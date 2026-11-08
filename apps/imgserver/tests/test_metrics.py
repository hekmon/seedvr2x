"""The server's /metrics text (metrics.py; DESIGN.md, The image server, /metrics), read back with
prometheus_client's parser, a dependency of the tests alone: every family the design lists with
its type, the values exact after known events, histograms with cumulative buckets (+Inf equal to
the count, the sum), a value on a bucket's bound counted in that bucket (le is inclusive), and
label values with a backslash, a double quote and a line feed surviving the round trip."""

import math
import threading

import pytest
from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from metrics import (
    CONTENT_TYPE,
    PREFIX,
    UPSCALE_BUCKETS,
    WAIT_BUCKETS,
    Gauges,
    Metrics,
)

INFO = {
    "version": "0.1.0",
    "dit_model": "seedvr2x_ema_7b_sharp_fp16.safetensors",
    "vae_model": "seedvr2x_ema_vae_fp16.safetensors",
    "attention": "sdpa",
}
GAUGES = Gauges(running=2, waiting=3, reserved=5 * 1024**3, reserved_peak=6 * 1024**3,
                reservation=7 * 1024**3, ready=True)  # fmt: skip


def read(metrics: Metrics, gauges: Gauges = GAUGES) -> dict[str, Metric]:
    return {
        family.name: family
        for family in text_string_to_metric_families(metrics.text(lambda: gauges))
    }


def samples(family: Metric, suffix: str = "") -> list[Sample]:
    return [each for each in family.samples if each.name == family.name + suffix]


def value(family: Metric) -> float:
    (only,) = family.samples
    return only.value


def named(name: str) -> str:
    return PREFIX + name


def test_the_content_type_is_the_text_formats() -> None:
    assert CONTENT_TYPE.startswith("text/plain; version=0.0.4")


def test_every_family_the_design_lists_is_there_with_its_type() -> None:
    families = read(Metrics(INFO))
    types = {name.removeprefix(PREFIX): family.type for name, family in families.items()}
    assert types == {
        "requests": "counter",
        "upscale_seconds": "histogram",
        "wait_seconds": "histogram",
        "output_pixels": "counter",
        "pictures_running": "gauge",
        "pictures_waiting": "gauge",
        "vram_reserved_bytes": "gauge",
        "vram_reserved_peak_bytes": "gauge",
        "vram_reservation_bytes": "gauge",
        "ready": "gauge",
        "gpu_wait_seconds": "gauge",
        "info": "gauge",
    }
    assert all(name.startswith(PREFIX) for name in families)


def test_every_family_has_a_help_line() -> None:
    families = read(Metrics(INFO))
    assert all(family.documentation for family in families.values())


def test_the_text_ends_with_a_line_feed_and_has_no_blank_line() -> None:
    text = Metrics(INFO).text(lambda: GAUGES)
    assert text.endswith("\n")
    assert "\n\n" not in text


def test_the_gauges_are_read_at_each_scrape() -> None:
    metrics = Metrics(INFO)
    read_at = []

    def gauges() -> Gauges:
        read_at.append(1)
        return GAUGES

    metrics.text(gauges)
    metrics.text(gauges)
    assert len(read_at) == 2


def test_the_gauges_read_back_exactly() -> None:
    families = read(Metrics(INFO))
    assert value(families[named("pictures_running")]) == 2
    assert value(families[named("pictures_waiting")]) == 3
    assert value(families[named("vram_reserved_bytes")]) == 5 * 1024**3
    assert value(families[named("vram_reserved_peak_bytes")]) == 6 * 1024**3
    assert value(families[named("vram_reservation_bytes")]) == 7 * 1024**3
    assert value(families[named("ready")]) == 1


def test_the_longest_wait_on_the_gpu_reads_back_in_seconds_and_is_zero_by_default() -> None:
    families = read(Metrics(INFO), Gauges(0, 0, 0, 0, 0, True, gpu_wait=12.5))
    assert value(families[named("gpu_wait_seconds")]) == 12.5
    assert value(read(Metrics(INFO))[named("gpu_wait_seconds")]) == 0


def test_ready_is_zero_when_not_ready() -> None:
    gauges = Gauges(0, 0, 0, 0, 0, False)
    families = read(Metrics(INFO), gauges)
    assert value(families[named("ready")]) == 0
    assert value(families[named("pictures_running")]) == 0


def test_a_server_that_served_nothing_has_the_families_and_zero_counts() -> None:
    families = read(Metrics(INFO))
    assert families[named("requests")].samples == []
    assert value(families[named("output_pixels")]) == 0
    for name in ("upscale_seconds", "wait_seconds"):
        family = families[named(name)]
        assert [each.value for each in samples(family, "_count")] == [0]
        assert [each.value for each in samples(family, "_sum")] == [0]
        assert all(each.value == 0 for each in samples(family, "_bucket"))


def test_the_requests_are_counted_by_status() -> None:
    metrics = Metrics(INFO)
    metrics.answered(200)
    metrics.answered(200)
    metrics.answered(503)
    family = read(metrics)[named("requests")]
    assert {each.labels["status"]: each.value for each in family.samples} == {
        "200": 2,
        "503": 1,
    }
    assert {each.name for each in family.samples} == {named("requests_total")}


def test_the_output_pixels_add_up() -> None:
    metrics = Metrics(INFO)
    metrics.upscaled(0.0, 1.0, 3840 * 2160)
    metrics.upscaled(0.0, 1.0, 1920 * 1080)
    assert value(read(metrics)[named("output_pixels")]) == 3840 * 2160 + 1920 * 1080


def test_a_histogram_is_cumulative_with_its_sum_and_count() -> None:
    metrics = Metrics(INFO)
    metrics.upscaled(0.0, 0.3, 1)  # <= 0.5
    metrics.upscaled(0.0, 1.5, 1)  # <= 2
    metrics.upscaled(0.0, 1.9, 1)  # <= 2
    metrics.upscaled(0.0, 1000.0, 1)  # past the last bound: +Inf only
    family = read(metrics)[named("upscale_seconds")]
    buckets = {each.labels["le"]: each.value for each in samples(family, "_bucket")}
    assert list(buckets) == [str(bound) for bound in UPSCALE_BUCKETS] + ["+Inf"]
    assert buckets["0.5"] == 1
    assert buckets["1.0"] == 1
    assert buckets["2.0"] == 3
    assert buckets["300.0"] == 3
    assert buckets["+Inf"] == 4
    values = list(buckets.values())
    assert values == sorted(values)
    assert [each.value for each in samples(family, "_count")] == [4]
    assert [each.value for each in samples(family, "_sum")] == [pytest.approx(1003.7)]
    assert buckets["+Inf"] == samples(family, "_count")[0].value


def test_the_wait_has_its_own_histogram() -> None:
    metrics = Metrics(INFO)
    metrics.upscaled(0.05, 3.0, 1)
    metrics.upscaled(0.0, 3.0, 1)
    families = read(metrics)
    wait = families[named("wait_seconds")]
    buckets = {each.labels["le"]: each.value for each in samples(wait, "_bucket")}
    assert list(buckets) == [str(bound) for bound in WAIT_BUCKETS] + ["+Inf"]
    assert buckets["0.01"] == 1
    assert buckets["0.1"] == 2
    assert [each.value for each in samples(wait, "_sum")] == [pytest.approx(0.05)]
    upscale = families[named("upscale_seconds")]
    assert [each.value for each in samples(upscale, "_count")] == [2]


@pytest.mark.parametrize("bound", UPSCALE_BUCKETS)
def test_a_value_on_a_bounds_is_in_that_bucket(bound: float) -> None:
    # le is inclusive: the bucket of le="0.5" counts 0.5.
    metrics = Metrics(INFO)
    metrics.upscaled(0.0, bound, 1)
    family = read(metrics)[named("upscale_seconds")]
    buckets = {float(each.labels["le"]): each.value for each in samples(family, "_bucket")}
    assert buckets[bound] == 1
    below = [each for each in UPSCALE_BUCKETS if each < bound]
    assert all(buckets[each] == 0 for each in below)


def test_a_value_just_past_a_bound_is_in_the_next() -> None:
    metrics = Metrics(INFO)
    metrics.upscaled(0.0, math.nextafter(0.5, 1.0), 1)
    buckets = {
        each.labels["le"]: each.value
        for each in samples(read(metrics)[named("upscale_seconds")], "_bucket")
    }
    assert buckets["0.5"] == 0
    assert buckets["1.0"] == 1


def test_the_info_carries_the_version_the_models_and_the_attention() -> None:
    family = read(Metrics(INFO))[named("info")]
    (sample,) = family.samples
    assert sample.labels == INFO
    assert sample.value == 1


def test_the_info_can_be_filled_after_the_start() -> None:
    metrics = Metrics()
    assert read(metrics)[named("info")].samples[0].labels == {}
    metrics.info.update(version="1", attention="sdpa")
    assert read(metrics)[named("info")].samples[0].labels == {
        "version": "1",
        "attention": "sdpa",
    }


@pytest.mark.parametrize(
    "text",
    ["back\\slash", 'a "quoted" name', "two\nlines", 'all \\ " \n three', "trailing\\", "é ü 日本"],
)
def test_label_values_survive_the_round_trip(text: str) -> None:
    family = read(Metrics({"dit_model": text, "version": "1"}))[named("info")]
    assert family.samples[0].labels["dit_model"] == text


def test_the_info_is_copied_at_the_start() -> None:
    given = {"version": "1"}
    metrics = Metrics(given)
    given["version"] = "2"
    assert read(metrics)[named("info")].samples[0].labels["version"] == "1"


def test_it_counts_from_several_threads() -> None:
    metrics = Metrics(INFO)

    def work() -> None:
        for _ in range(200):
            metrics.answered(200)
            metrics.upscaled(0.0, 1.0, 10)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    families = read(metrics)
    assert families[named("requests")].samples[0].value == 1600
    assert value(families[named("output_pixels")]) == 16000
    assert samples(families[named("upscale_seconds")], "_count")[0].value == 1600
