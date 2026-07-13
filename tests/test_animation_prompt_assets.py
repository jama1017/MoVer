import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONVERTER_ASSETS = ROOT / "src" / "mover" / "converter" / "assets"
COMPOSER_ASSETS = ROOT / "src" / "mover" / "composers" / "assets"
SYNTHESIZER_ASSETS = ROOT / "src" / "mover" / "synthesizers" / "assets"
UNBUNDLED_EASE_PLUGINS = ("CustomEase", "CustomBounce", "CustomWiggle")


class AnimationPromptAssetsTest(unittest.TestCase):
    def test_local_template_loads_documented_bundled_capabilities(self) -> None:
        template = (SYNTHESIZER_ASSETS / "template.html").read_text()

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
            CONVERTER_ASSETS / "api.js",
            COMPOSER_ASSETS / "template_animation_allow_gsap.md",
            SYNTHESIZER_ASSETS / "sys_msg_animation_synthesizer.md",
            SYNTHESIZER_ASSETS
            / "sys_msg_animation_synthesizer_with_implementation.md",
        )

        for path in paths:
            contents = path.read_text()
            for plugin in UNBUNDLED_EASE_PLUGINS:
                self.assertNotIn(plugin, contents, str(path))

        gsap_prompt = (
            SYNTHESIZER_ASSETS
            / "sys_msg_animation_synthesizer_with_implementation.md"
        ).read_text()
        self.assertNotIn("any other GSAP functions", gsap_prompt)
        self.assertIn(
            "Do not use other GSAP plugins unless the selected HTML template "
            "explicitly loads them.",
            gsap_prompt,
        )


if __name__ == "__main__":
    unittest.main()
