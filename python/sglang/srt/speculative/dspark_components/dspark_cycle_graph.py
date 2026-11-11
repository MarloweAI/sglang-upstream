"""One HIP graph a DSpark decode cycle (SGLANG_OPT_DSV41_DSPARK_CYCLE_GRAPH).

A static DSpark decode step replays the draft graph, the target verify graph and, on a
sampled step, the sampling-accept graph, with eager glue between them: the FutureMap
resolve, the verify window and draft anchor, the graphs' input copies and metadata prep.
On MI355X every graph launch costs ≈7 µs of GPU time beyond its kernels, and every glue
op ≈1.6–2.4 µs whether eager or captured.

This module records the scheduler's whole forward region of such a step (resolve, the
worker's decode step, the relay) as one CUDA graph: the glue ops are captured, and each
graph the step replays is spliced into the capture as flat node copies
(``graph_splice``). Later steps launch that graph once. The step's device state needs no
host input: seq lens and the bonus come from the FutureMap relay the verify graph
writes, cache locs from ``req_to_token`` (the scheduler allocates pages ahead of the
device lengths, as it does today), the request rows from this module's static index
buffer, sampling params from the draft sampler's static buffers. Host per step: the
simulated-acceptance draws (benchmark only, before the launch), the launch, the result
copy and ``process_batch_result``. With SGLANG_OPT_DSV41_DSPARK_CYCLE_RING the window
launch draws the simulated acceptance, and the result comes back through one copy into a
pinned ring the host polls (``dspark_cycle_ring``): no event a cycle.

Which batches it serves is decided from the batch alone (``DSparkWorkerV2.
cycle_graph_eligible`` and at most SGLANG_DSPARK_CYCLE_GRAPH_MAX_BS requests): like the
per-batch-size graphs, other batches take the per-stage path. One recording per batch
size, accept kind (greedy or sampled) and target attention variant (the candidate
indexer's, chosen from host lengths as the per-stage path chooses it), made after ``NUM_WARMUP`` per-stage steps of
that kind (they capture the lazily captured glue and accept graphs) and kept until a
spliced graph is captured again. A served step right after any other step (another
batch, a prefill) or of a new batch takes the per-stage path once: it restages what the
per-stage path caches by batch (the draft sampler's params, the static step state), and
what the recording reads from the batch is then the same.
"""

from __future__ import annotations

import copy
import enum
import logging
import time
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_utils.graph_splice import (
    describe_graph,
    graphs_epoch,
    splicing,
)
from sglang.srt.speculative.dspark_components.dspark_cycle_ring import (
    CycleResultRing,
    LazyPublishEvent,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)


class CycleRoute(enum.Enum):
    PER_STAGE = "per_stage"  # a served batch on today's path (warm-up, restage)
    RECORD = "record"
    REPLAY = "replay"


class _Recording:
    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        self.result: Optional["GenerationBatchResult"] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.rows: Optional[torch.Tensor] = None
        self.bonus: Optional[torch.Tensor] = None
        # The recorded window launch draws the simulated acceptance (cycle ring).
        self.device_draw = False
        self.splice: dict = {}
        self.t0 = time.perf_counter()


class DsparkCycleGraph:
    NUM_WARMUP = 3

    def __init__(self, *, worker, future_map, device) -> None:
        self.worker = worker
        self.future_map = future_map
        self.max_bs = int(envs.SGLANG_DSPARK_CYCLE_GRAPH_MAX_BS.get())
        self._pool = torch.cuda.graph_pool_handle()
        # Not the forward stream: a stream that is capturing fails every query of an
        # event last recorded on it (the NCCL watchdog's of the prefill's collectives).
        self._capture_stream = torch.cuda.Stream()
        # The request rows every recording reads (FutureMap rows, req_to_token rows).
        self._rows = torch.zeros((self.max_bs,), dtype=torch.int64, device=device)
        self._rows_of: Optional[torch.Tensor] = None  # whose rows the rows buffer holds
        self._rows_in: Optional[torch.Tensor] = None
        self._recordings: Dict[tuple, _Recording] = {}
        self._warm: Dict[tuple, int] = {}
        self._epoch = graphs_epoch()
        # The previous step's batch, when it was a served step of the same batch.
        self._last: Optional[tuple] = None
        self._recording: Optional[_Recording] = None
        self._key: Optional[tuple] = None
        self._batch_rows: Optional[torch.Tensor] = None
        self.num_recordings = 0
        self.num_replays = 0
        # SGLANG_OPT_DSV41_DSPARK_CYCLE_RING: replays return their result through a
        # pinned ring and mark the publish event instead of recording it.
        self.ring = (
            CycleResultRing() if envs.SGLANG_OPT_DSV41_DSPARK_CYCLE_RING.get() else None
        )

    # ---- routing ---------------------------------------------------------------

    @staticmethod
    def _identity(batch: "ScheduleBatch") -> tuple:
        sampling = batch.sampling_info
        # Strong references: the ids can't be reused while held.
        return (batch.req_pool_indices, sampling, len(batch.reqs))

    def route_none(self) -> None:
        """A step the cycle graph doesn't serve."""
        self._last = None
        return None

    def route(self, batch: "ScheduleBatch") -> Optional[CycleRoute]:
        """This step's path; None for a batch the cycle graph doesn't serve."""
        bs = len(batch.reqs)
        if bs > self.max_bs or not self.worker.cycle_graph_eligible(batch):
            return self.route_none()
        if self._epoch != graphs_epoch():
            # A spliced graph was captured again: its copies are stale.
            for rec in self._recordings.values():
                rec.graph.reset()
            self._recordings.clear()
            # A pool whose graphs are all gone can't take another capture.
            self._pool = torch.cuda.graph_pool_handle()
            self._warm.clear()
            self._epoch = graphs_epoch()
        key = (
            bs,
            self.worker.cycle_graph_greedy(batch),
            self.worker.cycle_graph_variant(batch),
        )
        identity = self._identity(batch)
        last, self._last = self._last, identity
        same = last is not None and all(x is y for x, y in zip(last, identity))
        self._key = key
        if key in self._recordings:
            return CycleRoute.REPLAY if same else CycleRoute.PER_STAGE
        warm = self._warm.get(key, 0)
        if warm < self.NUM_WARMUP or not same:
            self._warm[key] = warm + 1
            return CycleRoute.PER_STAGE
        return CycleRoute.RECORD

    def _rows_buffer(self) -> torch.Tensor:
        """The graph runners' shared req_pool_indices slot when there is one (their
        fills then copy nothing), else this module's buffer."""
        slot = self.worker._target_input_slot("req_pool_indices")
        if slot is not None and slot.dtype == torch.int64 and slot.numel() >= self.max_bs:
            return slot
        return self._rows

    def _bind_rows(self, batch: "ScheduleBatch") -> torch.Tensor:
        """This batch's request rows in the rows buffer (a copy when they changed)."""
        rows = self._rows_buffer()
        bs = len(batch.reqs)
        if self._rows_of is not batch.req_pool_indices or self._rows_in is not rows:
            rows[:bs].copy_(batch.req_pool_indices)
            self._rows_of, self._rows_in = batch.req_pool_indices, rows
        return rows[:bs]

    # ---- recording -------------------------------------------------------------

    def begin_record(self, batch: "ScheduleBatch") -> None:
        """Start capturing the step's forward region on the capture stream, which is
        current until ``end_record``; the batch reads its rows from ``_rows`` meanwhile."""
        assert self._recording is None
        rows = self._bind_rows(batch)
        self.worker.begin_cycle_step(batch)
        rec = _Recording(graphs_epoch())
        rec.rows = rows
        self._recording = rec
        self._batch_rows = batch.req_pool_indices
        batch.req_pool_indices = rows
        batch.spec_info.future_indices = rows
        self._stream_ctx = torch.cuda.stream(self._capture_stream)
        self._stream_ctx.__enter__()
        rec.graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")
        self._splice_ctx = splicing()
        rec.splice = self._splice_ctx.__enter__()

    def end_record(
        self, batch: "ScheduleBatch", result: "GenerationBatchResult"
    ) -> "GenerationBatchResult":
        """End the capture, run this step through the recorded graph, keep the result's
        device tensors as the graph's outputs."""
        rec, self._recording = self._recording, None
        self._splice_ctx.__exit__(None, None, None)
        rec.graph.capture_end()
        self._stream_ctx.__exit__(None, None, None)
        batch.req_pool_indices = self._batch_rows
        self._batch_rows = None
        self.worker.end_cycle_step()
        # Events recorded in the capture are never recorded on a stream.
        self.worker.model_runner.shared_read_done_event = None
        if not self.worker.last_step_graph_accept:
            raise RuntimeError(
                "DSpark cycle graph: a served step took the eager accept; "
                "cycle_graph_eligible disagrees with the step's routing"
            )
        if rec.epoch != graphs_epoch():
            raise RuntimeError("DSpark cycle graph: a graph was captured while recording")
        if self.worker.last_verify_variant() != self._key[2]:
            raise RuntimeError(
                "DSpark cycle graph: the step replayed verify variant %r, predicted %r"
                % (self.worker.last_verify_variant(), self._key[2])
            )
        rec.result = self._snapshot(result)
        rec.device_draw = self.worker.last_step_device_draw
        if self.ring is not None:
            CycleResultRing.check(rec.result)
        rec.seq_lens = batch.seq_lens
        rec.bonus = getattr(batch.spec_info, "bonus_tokens", None)
        nodes = describe_graph(rec.graph.raw_cuda_graph())
        t1 = time.perf_counter()
        rec.graph.replay()  # instantiates, then runs this step
        self._recordings[self._key] = rec
        self.num_recordings += 1
        logger.info(
            "DSpark cycle graph recorded for %s: %d graphs spliced (%d nodes), nodes %s, "
            "capture %.1f ms, instantiate and launch %.1f ms.",
            "bs %d %s %s" % (self._key[0], "greedy" if self._key[1] else "sampled", self._key[2]),
            rec.splice.get("graphs", 0),
            rec.splice.get("nodes", 0),
            nodes,
            (t1 - rec.t0) * 1e3,
            (time.perf_counter() - t1) * 1e3,
        )
        return result

    # ---- replay ----------------------------------------------------------------

    def replay(self, batch: "ScheduleBatch") -> "GenerationBatchResult":
        rec = self._recordings[self._key]
        rows = self._bind_rows(batch)
        assert rows.data_ptr() == rec.rows.data_ptr(), "the recording's rows buffer moved"
        self.worker.begin_cycle_step(batch, device_draw=rec.device_draw)
        rec.graph.replay()
        self.worker.end_cycle_step()
        self.num_replays += 1
        # The host-visible state the forward region leaves behind.
        batch.seq_lens = rec.seq_lens
        batch.seq_lens_cpu = None
        batch.seq_lens_sum = None
        if rec.bonus is not None and batch.spec_info is not None:
            batch.spec_info.bonus_tokens = rec.bonus
        out = self._snapshot(rec.result)
        if self.ring is not None:
            fm = self.future_map
            if not isinstance(fm.publish_ready, LazyPublishEvent):
                fm.publish_ready = LazyPublishEvent()
            fm.publish_ready.mark(torch.cuda.current_stream())
            self.ring.issue(out)
        elif self.future_map.publish_ready is not None:
            self.future_map.publish_ready.record()
        self.worker.model_runner.shared_read_done_event = None
        return out

    @staticmethod
    def _snapshot(result: "GenerationBatchResult") -> "GenerationBatchResult":
        """A per-step copy of a result holding the graph's output tensors (the scheduler
        replaces fields with host copies)."""
        out = copy.copy(result)
        if result.logits_output is not None:
            out.logits_output = copy.copy(result.logits_output)
        if result.next_draft_input is not None:
            out.next_draft_input = copy.copy(result.next_draft_input)
        out.copy_done = None
        return out
