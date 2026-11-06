"""Headless smoke test of the Streamlit page, using Streamlit's own AppTest.

This drives the real page: it fills widgets, runs the pipeline (real easyocr)
and inspects the rendered element tree. No browser, no websocket.

It takes ~30 s because OCR is real, so it is skipped by default:

    MENU_E2E=1 python -m unittest tests.test_streamlit_app -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

try:
    from streamlit.testing.v1 import AppTest
except Exception:  # streamlit missing or too old for the testing API
    AppTest = None  # type: ignore[assignment]

TABS = ["Plan", "Dishes", "Price table", "Classification", "Behind the scenes"]

# Only the sidebar picker and the landing gallery - no OCR, no model call, so
# this reproduces the widget-state wiring in well under a second.
PICKER_SCRIPT = f"""
import sys
sys.path.insert(0, {str(PROJECT_DIR)!r})
import streamlit as st
import streamlit_app as app
app._image_choice()
app._landing()
"""

# The taste sidebar on its own: widget seeding, saving, and the memory footer.
# MENU_DATA_DIR is set by the test, so this never touches the real profile.
TABLE_SCRIPT = f"""
import sys
sys.path.insert(0, {str(PROJECT_DIR)!r})
import streamlit as st
import streamlit_app as app
app._table_sidebar()
"""


@unittest.skipUnless(AppTest is not None, "streamlit is not installed")
class LandingPickerTests(unittest.TestCase):
    """The gallery buttons drive the sidebar selectbox through session state."""

    def setUp(self) -> None:
        # Imported here, not at module level: streamlit_app calls st.html at
        # import time, which only makes sense inside a script run.
        from streamlit_app import SAMPLES

        self.samples = [name for name in SAMPLES if (PROJECT_DIR / name).exists()]
        self.at = AppTest.from_string(PICKER_SCRIPT, default_timeout=120)
        self.at.run()
        self.assertFalse(self.at.exception)

    def test_landing_starts_with_no_sample_selected(self):
        self.assertEqual(self.at.selectbox[0].value, "(none)")
        self.assertEqual(
            [button.label for button in self.at.button], ["Scan this"] * len(self.samples)
        )

    def test_gallery_button_selects_that_sample(self):
        """Regression: writing the widget key from the page body raised
        StreamlitAPIException ("cannot be modified after the widget with key
        `sample_choice` is instantiated"). It must go through a callback."""
        self.at.button[2].click().run()
        self.assertFalse(self.at.exception, [exc.message for exc in self.at.exception])
        expected = self.samples[2]
        self.assertEqual(self.at.session_state["sample_choice"], expected)
        self.assertEqual(self.at.selectbox[0].value, expected)

    def test_every_gallery_button_works(self):
        for index, name in enumerate(self.samples):
            self.at.button[index].click().run()
            self.assertFalse(self.at.exception, [exc.message for exc in self.at.exception])
            self.assertEqual(self.at.session_state["sample_choice"], name)


@unittest.skipUnless(AppTest is not None, "streamlit is not installed")
class TableSidebarTests(unittest.TestCase):
    """The taste sidebar: seeding from disk, saving back, and the memory footer."""

    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self._previous = os.environ.get("MENU_DATA_DIR")
        os.environ["MENU_DATA_DIR"] = self.dir
        self.at = AppTest.from_string(TABLE_SCRIPT, default_timeout=60)
        self.at.run()
        self.assertFalse(self.at.exception, [e.message for e in self.at.exception])

    def tearDown(self) -> None:
        if self._previous is None:
            os.environ.pop("MENU_DATA_DIR", None)
        else:
            os.environ["MENU_DATA_DIR"] = self._previous

    def test_asks_for_the_four_things_the_user_knows(self):
        labels = [box.label for box in self.at.number_input]
        self.assertEqual(labels, ["How many people?", "Total budget (CNY, 0 = no limit)"])
        self.assertEqual(len(self.at.multiselect), 1)   # allergens
        self.assertEqual(len(self.at.checkbox), 2)      # learn spice, vegetarian only
        self.assertTrue(any("Anything else" in box.label for box in self.at.text_area))

    def test_defaults_come_from_the_profile_on_disk(self):
        self.assertEqual(self.at.number_input[0].value, 2)

    def test_a_change_is_written_to_the_profile_file(self):
        self.at.number_input[0].set_value(5).run()
        self.assertFalse(self.at.exception, [e.message for e in self.at.exception])
        saved = json.loads((Path(self.dir) / "profile.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["party_size"], 5)

    def test_allergens_are_saved_as_a_tuple(self):
        self.at.multiselect[0].set_value(["peanut", "sesame"]).run()
        saved = json.loads((Path(self.dir) / "profile.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["avoid_allergens"], ["peanut", "sesame"])

    def test_memory_footer_says_nothing_is_learned_yet(self):
        self.assertTrue(
            any("Nothing learned yet" in caption.value for caption in self.at.caption)
        )


@unittest.skipUnless(
    AppTest is not None and os.environ.get("MENU_E2E"),
    "set MENU_E2E=1 to run the app test (drives real OCR, ~30 s)",
)
class StreamlitAppTests(unittest.TestCase):
    def _app(self) -> "AppTest":
        return AppTest.from_file(str(PROJECT_DIR / "streamlit_app.py"), default_timeout=300)

    def test_landing_page_offers_the_samples(self):
        from streamlit_app import SAMPLES

        at = self._app()
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Menu Vision" in block.value for block in at.markdown))
        self.assertEqual([box.label for box in at.selectbox], ["or use a sample image"])
        self.assertEqual(len(at.tabs), 0, "results tabs should not exist before a scan")
        expected = sum(1 for name in SAMPLES if (PROJECT_DIR / name).exists())
        self.assertEqual(len(at.button), expected, "one 'Scan this' button per sample")

    def test_scan_without_the_model_lists_every_line(self):
        at = self._app()
        at.run()
        # The OCR reader is chosen explicitly: with no model call this test must
        # not need an API key, and the vision reader always needs one.
        at.segmented_control[0].set_value("ocr").run()
        at.toggle[0].set_value(False).run()  # translation off: OCR + layout only
        at.selectbox[0].set_value("test.jpg").run()

        self.assertFalse(at.exception, [exc.message for exc in at.exception])
        self.assertEqual([tab.label for tab in at.tabs], TABS)
        metrics = {metric.label: metric.value for metric in at.metric}
        self.assertGreaterEqual(int(metrics["Dishes"]), 1)
        # Nothing classifies UI lines away when the model is skipped.
        self.assertEqual(metrics["Rejected"], "0")
        self.assertTrue(len(at.dataframe) >= 1)
        self.assertTrue(any("No language model was used" in info.value for info in at.info))

    def test_partner_settings_do_not_trigger_a_rescan(self):
        """Changing the party or the budget is a ranking change, not a new read.

        The scan key must stay put - re-running OCR because somebody moved a
        slider would cost 40-120 seconds of their dinner.
        """
        at = self._app()
        at.run()
        at.segmented_control[0].set_value("ocr").run()
        at.toggle[0].set_value(False).run()
        at.selectbox[0].set_value("test.jpg").run()
        self.assertFalse(at.exception)

        before_key = at.session_state["scan"]["key"]
        before_plan = {
            metric.label: metric.value for metric in at.metric
        }["To order"]

        at.number_input[0].set_value(6).run()  # party size
        self.assertFalse(at.exception, [e.message for e in at.exception])

        self.assertEqual(at.session_state["scan"]["key"], before_key, "re-scanned")
        after_plan = {metric.label: metric.value for metric in at.metric}["To order"]
        self.assertGreaterEqual(int(after_plan), int(before_plan))

    def test_the_plan_never_shows_a_total_for_an_unpriced_menu(self):
        at = self._app()
        at.run()
        at.segmented_control[0].set_value("ocr").run()
        at.toggle[0].set_value(False).run()
        at.selectbox[0].set_value("test2.jpg").run()  # skewers, no prices printed
        self.assertFalse(at.exception, [e.message for e in at.exception])
        self.assertEqual(
            {metric.label: metric.value for metric in at.metric}["Total"], "unknown"
        )


if __name__ == "__main__":
    unittest.main()
