import unittest
from importlib import resources, util
from unittest.mock import patch

from mover._optional import require_modules
from mover.synthesizers.llm_client import LLMClient


CONVERTER_ASSETS = resources.files("mover.converter").joinpath("assets")
DSL_ASSETS = resources.files("mover.dsl").joinpath("assets")
NLG_ASSETS = resources.files("mover.nlg").joinpath("assets")
SYNTHESIZER_ASSETS = resources.files("mover.synthesizers").joinpath("assets")
UNBUNDLED_EASE_PLUGINS = ("CustomEase", "CustomBounce", "CustomWiggle")
REQUIRED_CONVERTER_ASSETS = {
    "EasePack.min.js",
    "MotionPathPlugin.min.js",
    "api.js",
    "convert.js",
    "grid.svg",
    "gsap.min.js",
    "index.css",
    "property_registry.json",
    "vis.js",
}


class AnimationPromptAssetsTest(unittest.TestCase):
    def test_local_template_loads_documented_bundled_capabilities(self) -> None:
        template = SYNTHESIZER_ASSETS.joinpath("template.html").read_text(
            encoding="utf-8"
        )

        for asset in (
            "gsap.min.js",
            "MotionPathPlugin.min.js",
            "EasePack.min.js",
            "api.js",
        ):
            self.assertIn(f'src="./{asset}"', template)
        for plugin in UNBUNDLED_EASE_PLUGINS:
            self.assertNotIn(plugin, template)

    def test_bundled_prompts_do_not_advertise_unloaded_plugins(self) -> None:
        paths = (
            CONVERTER_ASSETS.joinpath("api.js"),
            SYNTHESIZER_ASSETS.joinpath("sys_msg_animation_synthesizer.md"),
            SYNTHESIZER_ASSETS.joinpath(
                "sys_msg_animation_synthesizer_with_implementation.md"
            ),
        )

        for path in paths:
            contents = path.read_text(encoding="utf-8")
            for plugin in UNBUNDLED_EASE_PLUGINS:
                self.assertNotIn(plugin, contents, str(path))

        gsap_prompt = SYNTHESIZER_ASSETS.joinpath(
            "sys_msg_animation_synthesizer_with_implementation.md"
        ).read_text(encoding="utf-8")
        self.assertNotIn("any other GSAP functions", gsap_prompt)
        self.assertIn(
            "Do not use other GSAP plugins unless the selected HTML template "
            "explicitly loads them.",
            gsap_prompt,
        )

    def test_installed_package_exposes_every_runtime_asset(self) -> None:
        for asset_name in REQUIRED_CONVERTER_ASSETS:
            asset = CONVERTER_ASSETS.joinpath(asset_name)
            self.assertTrue(asset.is_file(), asset_name)
        self.assertFalse(CONVERTER_ASSETS.joinpath("library.js").is_file())
        self.assertFalse(CONVERTER_ASSETS.joinpath("library_raw.js").is_file())

        for package_assets, names in (
            (DSL_ASSETS, ("correction_msg_template.md",)),
            (NLG_ASSETS, ("sentence_patterns.json", "vocab.json")),
            (
                SYNTHESIZER_ASSETS,
                (
                    "sys_msg_animation_synthesizer.md",
                    "sys_msg_animation_synthesizer_with_implementation.md",
                    "sys_msg_mover_synthesizer.md",
                    "sys_msg_prompt_rewriter.md",
                    "template.html",
                ),
            ),
        ):
            for asset_name in names:
                self.assertTrue(
                    package_assets.joinpath(asset_name).is_file(),
                    asset_name,
                )

        vendor_root = resources.files("mover._vendor")
        for package_name in ("concepts", "torch_index"):
            package = vendor_root.joinpath(package_name)
            self.assertTrue(package.joinpath("LICENSE").is_file())
            self.assertTrue(package.joinpath("VENDORED.md").is_file())

        package_root = resources.files("mover")
        if "site-packages" in str(package_root):
            self.assertFalse(
                SYNTHESIZER_ASSETS.joinpath("sys_msg_test.md").is_file()
            )
            self.assertIsNone(util.find_spec("mover.composers"))

    def test_full_and_provider_errors_name_the_install_extra(self) -> None:
        with patch("mover._optional.is_module_available", return_value=False):
            with self.assertRaisesRegex(
                ModuleNotFoundError,
                r'pip install "mover\[full\]"',
            ):
                require_modules(
                    extra="full",
                    feature="The MoVer generation pipeline",
                    modules={"torch": "torch"},
                )

        missing_groq = ModuleNotFoundError(
            "No module named 'groq'",
            name="groq",
        )
        with patch(
            "mover._optional.importlib.import_module",
            side_effect=missing_groq,
        ):
            with self.assertRaisesRegex(
                ModuleNotFoundError,
                r'pip install "mover\[groq\]"',
            ):
                LLMClient("test", "groq")


if __name__ == "__main__":
    unittest.main()
