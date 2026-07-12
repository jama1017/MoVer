import io
import json
import asyncio
import argparse
import tempfile
import shutil
from typing import List
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright, Page
import subprocess
from PIL import Image
import uvicorn


VIDEO_OUTPUT_FORMATS = {"mp4", "gif"}
FRAME_OUTPUT_FORMATS = {"png", "svg"}
SUPPORTED_OUTPUT_FORMATS = VIDEO_OUTPUT_FORMATS | FRAME_OUTPUT_FORMATS
HIDE_GRID_SCREENSHOT_STYLE = "svg { background-image: none !important; }"


def _get_animation_output_path(
    output_dir: str | Path,
    base_name: str,
    output_format: str,
    fps: int,
) -> Path:
    """Build an output path while preserving established naming contracts."""
    normalized_format = output_format.lower()
    if normalized_format in FRAME_OUTPUT_FORMATS:
        return Path(output_dir) / f"{base_name}_animation_{fps}_{normalized_format}"
    if normalized_format in VIDEO_OUTPUT_FORMATS:
        return Path(output_dir) / f"{base_name}_animation.{normalized_format}"
    raise ValueError(f"Unsupported output format: {normalized_format}")


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    output_format: str = "mp4",
) -> None:
    """Create MP4/GIF output; GIF requires FFmpeg, MP4 can fall back to OpenCV."""
    if not frames:
        raise ValueError("No frames provided")
    if output_format not in VIDEO_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported video output format: {output_format}")

    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        ffmpeg_available = True
    except (subprocess.SubprocessError, FileNotFoundError):
        ffmpeg_available = False

    if output_format == "gif" and not ffmpeg_available:
        raise RuntimeError("GIF output requires a working FFmpeg installation")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    ## Always create MP4 first
    temp_mp4_path = Path(output_path).with_suffix('.temp.mp4')
    out = cv2.VideoWriter(str(temp_mp4_path), fourcc, fps, (width, height))

    try:
        for frame in frames:
            out.write(frame)
    finally:
        out.release()

    if ffmpeg_available:
        try:
            if output_format == "gif":
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', str(temp_mp4_path),
                    '-vf', f'fps={fps},split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5',
                    str(output_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                ## Convert to web-compatible MP4
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', str(temp_mp4_path),
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    str(output_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            temp_mp4_path.unlink()
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            if output_format == "gif":
                temp_mp4_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "GIF output requires FFmpeg with GIF palette support"
                )
            print("ffmpeg conversion failed, using OpenCV MP4 output")

    if temp_mp4_path.exists():
        temp_mp4_path.replace(output_path)


async def capture_frames_server_driven(
    page: Page,
    output_path: str,
    fps: int = 30,
    output_format: str = "mp4",
    in_memory: bool = False,
    hide_grid: bool = False,
) -> None | tuple[list[np.ndarray | io.StringIO], float]:
    """
    Server-driven frame capture: Python controls the timeline and screenshots.
    Video/GIF frames are streamed through a temp directory; PNG/SVG outputs
    are saved as per-frame files in a directory. In-memory PNG/SVG capture
    returns ``(frames, duration)`` without creating, modifying, or deleting
    filesystem output. ``hide_grid`` suppresses the SVG grid only in raster
    screenshots and does not mutate the page's persistent styles. GIF output
    requires a working FFmpeg installation.
    """
    output_format = output_format.lower()
    if fps <= 0:
        raise ValueError("fps must be positive")
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")
    if in_memory and output_format not in FRAME_OUTPUT_FORMATS:
        raise ValueError("in_memory=True is only supported for PNG and SVG output")

    ## Get animation info from the page using the same FPS used for seeking.
    anim_info = await page.evaluate("fps => getAnimationInfo(fps)", fps)
    duration = float(anim_info["animDuration"])
    total_frames = anim_info["steps"]
    capture_frame_count = total_frames + 1

    print(f"Capturing {capture_frame_count} frames at {fps} FPS (duration: {duration}s)")

    ## Hide GSDevTools if present (it overlays on top of the SVG).
    await page.evaluate("""() => {
        const devtools = document.querySelector('#GSDevTools');
        if (devtools) devtools.style.display = 'none';
        // Also hide any GSDevTools container elements
        document.querySelectorAll('[class*="gs-dev-tools"]').forEach(el => el.style.display = 'none');
        // Stop wall-clock advancement. seekToFrame maps requested local times
        // through tl_to_use.globalTime(...) so sibling GSAP animations advance
        // coherently when the root timeline is available.
        if (typeof gsap !== "undefined" && gsap.globalTimeline) {
            gsap.globalTimeline.pause();
        }
    }""")

    ## Locate the SVG element once.
    svg_element = page.locator("svg").first
    await svg_element.wait_for(state="visible")

    in_memory_frames: list[np.ndarray | io.StringIO] = []
    frames_dir: Path | None = None
    cleanup_temp_dir = False
    if not in_memory:
        output_target = Path(output_path)
        cleanup_temp_dir = output_format in VIDEO_OUTPUT_FORMATS
        if cleanup_temp_dir:
            output_target.parent.mkdir(parents=True, exist_ok=True)
            frames_dir = Path(tempfile.mkdtemp(prefix="mover_frames_"))
        else:
            frames_dir = output_target
            if frames_dir.suffix.lower() == f".{output_format}":
                frames_dir = frames_dir.with_suffix("")
            if frames_dir.exists() and not frames_dir.is_dir():
                raise ValueError(f"Frame output path exists and is not a directory: {frames_dir}")
            frames_dir.mkdir(parents=True, exist_ok=True)
            for existing_frame in frames_dir.glob(f"frame_*.{output_format}"):
                existing_frame.unlink()

    try:
        for frame_index in range(capture_frame_count):
            await page.evaluate(f"() => seekToFrame({frame_index}, {fps}, {duration})")

            await page.evaluate(
                "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
            )

            if output_format == "svg":
                svg_markup = await page.evaluate("""() => {
                    const svg = document.querySelector("svg");
                    if (!svg) throw new Error("No SVG element found");
                    return new XMLSerializer().serializeToString(svg);
                }""")
                if in_memory:
                    in_memory_frames.append(io.StringIO(f"{svg_markup}\n"))
                else:
                    assert frames_dir is not None
                    frame_path = frames_dir / f"frame_{frame_index:06d}.svg"
                    frame_path.write_text(f"{svg_markup}\n", encoding="utf-8")
            else:
                if hide_grid:
                    png_bytes = await svg_element.screenshot(
                        type="png",
                        style=HIDE_GRID_SCREENSHOT_STYLE,
                    )
                else:
                    png_bytes = await svg_element.screenshot(type="png")

                if in_memory:
                    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                    in_memory_frames.append(np.asarray(img, dtype=np.float32) / 255.0)
                else:
                    assert frames_dir is not None
                    frame_path = frames_dir / f"frame_{frame_index:06d}.png"
                    with open(frame_path, "wb") as f:
                        f.write(png_bytes)

            if (frame_index + 1) % max(1, capture_frame_count // 10) == 0 or frame_index == capture_frame_count - 1:
                print(f"  Captured frame {frame_index + 1}/{capture_frame_count}")

        if in_memory:
            return in_memory_frames, duration

        if output_format in FRAME_OUTPUT_FORMATS:
            assert frames_dir is not None
            print(f"{output_format.upper()} frames saved to {frames_dir}")
            return

        ## Encode video using FFmpeg directly from image sequence.
        assert frames_dir is not None
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)

            if output_format == "gif":
                subprocess.run([
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', str(frames_dir / 'frame_%06d.png'),
                    '-vf', f'fps={fps},split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5',
                    str(output_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                subprocess.run([
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', str(frames_dir / 'frame_%06d.png'),
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    str(output_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            print(f"Video saved to {output_path}")

        except (subprocess.SubprocessError, FileNotFoundError) as error:
            if output_format == "gif":
                raise RuntimeError(
                    "GIF output requires a working FFmpeg installation"
                ) from error
            print("ffmpeg not found, falling back to OpenCV encoding")
            # Fallback: read frames back and use OpenCV
            video_frames = []
            for frame_index in range(capture_frame_count):
                frame_path = frames_dir / f"frame_{frame_index:06d}.png"
                img = Image.open(frame_path)
                if img.mode != 'RGB':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[3])
                    else:
                        background.paste(img)
                    img = background
                video_frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            create_video_from_frames(video_frames, output_path, fps, output_format)
            print(f"Video saved to {output_path} (OpenCV fallback)")

    finally:
        if cleanup_temp_dir and frames_dir is not None:
            shutil.rmtree(frames_dir, ignore_errors=True)


def setup_fastapi_app(html_file: str, html_dir: str, base_name: str, output_format: str = "mp4", output_dir: str | None = None, save_animated_properties: bool = False) -> FastAPI:
    """Set up and configure the FastAPI application."""
    out_dir = output_dir or html_dir
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/convert-js-to-json")
    async def convert_js_to_json(request: Request):
        """Convert JavaScript data to JSON and save it."""
        json_data = await request.json()
        json_file_path = Path(out_dir) / f"{base_name}_data.json"
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print("SAVED TO LOCAL")
        return JSONResponse(content={"status": "success"})
    
    @app.post("/convert-js-to-keyframes-json")
    async def convert_js_to_keyframes_json(request: Request):
        """Convert JavaScript keyframes data to JSON and save it."""
        json_data = await request.json()
        json_file_path = Path(out_dir) / f"{base_name}_data_keyframes.json"
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print("SAVED KEYFRAMES TO LOCAL")
        return JSONResponse(content={"status": "success"})

    @app.post("/convert-js-to-rendered-json")
    async def convert_js_to_rendered_json(request: Request):
        """Convert JavaScript rendered comparison data to JSON and save it."""
        json_data = await request.json()
        json_file_path = Path(out_dir) / f"{base_name}_data_rendered.json"
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print("SAVED RENDERED DATA TO LOCAL")
        return JSONResponse(content={"status": "success"})

    @app.post("/save-animated-properties")
    async def save_animated_properties_endpoint(request: Request):
        """Save extracted animated properties (registry names per element) to JSON."""
        json_data = await request.json()
        json_file_path = Path(out_dir) / f"{base_name}_properties.json"
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"SAVED ANIMATED PROPERTIES TO {json_file_path}")
        return JSONResponse(content={"status": "success"})

    @app.get("/")
    async def serve_html():
        """Serve the HTML file."""
        return FileResponse(html_file)

    # Mount the assets directory at root to allow relative path access
    assets_path = Path(__file__).parent / "assets"
    app.mount("/", StaticFiles(directory=str(assets_path), follow_symlink=True), name="assets")

    return app


def handle_console_message(msg, print_console: bool):
    """Handle console messages from the browser page."""
    if print_console:
        msg_type = msg.type
        location = msg.location
        location_str = ""
        if location:
            url = location.get('url', 'unknown')
            line = location.get('lineNumber', '?')
            col = location.get('columnNumber', '?')
            location_str = f" (at {url}:{line}:{col})"
        print(f"    [JS Console {msg_type.upper()}] {msg.text}{location_str}")


def handle_network_response(response, print_console: bool):
    """Handle network responses from the browser page."""
    if print_console:
        if response.status >= 400:
            print(f"    [Network Error] {response.status} {response.status_text}: {response.url}")


async def _wait_for_server_start(server: uvicorn.Server, server_task: asyncio.Task, timeout_s: float = 10.0) -> None:
    """Wait until Uvicorn has bound its socket and completed startup."""
    start_time = asyncio.get_running_loop().time()
    while not server.started:
        if server_task.done():
            await server_task
        if asyncio.get_running_loop().time() - start_time > timeout_s:
            raise TimeoutError("Timed out waiting for conversion server to start")
        await asyncio.sleep(0.01)


def _get_bound_port(server: uvicorn.Server) -> int:
    """Return the actual localhost port bound by Uvicorn."""
    if not server.servers:
        raise RuntimeError("Conversion server did not expose any bound sockets")
    sockets = server.servers[0].sockets
    if not sockets:
        raise RuntimeError("Conversion server did not expose any bound sockets")
    return sockets[0].getsockname()[1]


async def run_conversion(html_file: str, port: int, create_video: bool = False, disable_easing: bool = False, save_keyframes: bool = False, save_for_comparison: bool = False, output_format: str = "mp4", video_fps: int = 30, print_console: bool = False, comparison_properties: dict | None = None, output_dir: str | None = None, save_animated_properties: bool = False, hide_grid: bool = False) -> None:
    """Run the conversion process."""
    html_path = Path(html_file)
    html_dir = str(html_path.parent)
    base_name = html_path.stem
    
    ## Ensure output_dir exists if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    app = setup_fastapi_app(html_file, html_dir, base_name, output_format, output_dir, save_animated_properties)

    # Configure uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    
    # Start the server as a task
    server_task = asyncio.create_task(server.serve())

    try:
        await _wait_for_server_start(server, server_task)
        actual_port = _get_bound_port(server)

        # Initialize Playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # Set up console logging and network error handlers
            page.on("console", lambda msg: handle_console_message(msg, print_console))
            page.on("response", lambda response: handle_network_response(response, print_console))

            print("Page loading time: ", end="", flush=True)
            start_time = asyncio.get_event_loop().time()
            ## Use networkidle to wait for all resources (including web fonts) to load
            await page.goto(f"http://127.0.0.1:{actual_port}", wait_until="networkidle")
            ## Explicitly wait for all fonts to be ready
            await page.evaluate("document.fonts.ready")
            print("Fonts loaded")
            load_time = asyncio.get_event_loop().time() - start_time
            print(f"{load_time:.2f} seconds")

            await capture_json_animation(actual_port, comparison_properties, disable_easing, page,
                                         save_animated_properties, save_for_comparison, save_keyframes, video_fps)

            ## Server-driven animation output — no HTTP round-trips per frame.
            if create_video:
                output_format = output_format.lower()
                if output_format not in SUPPORTED_OUTPUT_FORMATS:
                    raise ValueError(f"Unsupported output format: {output_format}")
                output_label = f"{output_format.upper()} frames" if output_format in FRAME_OUTPUT_FORMATS else output_format.upper()
                print(f"Creating {output_label}...")
                video_out_dir = output_dir or html_dir
                output_path = _get_animation_output_path(
                    video_out_dir,
                    base_name,
                    output_format,
                    video_fps,
                )
                await capture_frames_server_driven(
                    page,
                    str(output_path),
                    video_fps,
                    output_format,
                    hide_grid=hide_grid,
                )

            await browser.close()
            
    finally:
        # Stop the server
        server.should_exit = True
        await server.shutdown()
        if not server_task.done():
            await server_task

async def capture_json_animation(actual_port: int, comparison_properties: dict | None, disable_easing: bool, page: Page,
                                 save_animated_properties: bool, save_for_comparison: bool, save_keyframes: bool,
                                 video_fps: int):
    ## Execute JavaScript in the page context
    registry = None
    if save_for_comparison or save_animated_properties:
        registry_path = Path(__file__).parent / "assets" / "property_registry.json"
        with open(registry_path) as f:
            registry = json.load(f)
    await page.evaluate(
        """([port, disableEasing, saveKeyframes, saveForComparison, registry, comparisonProperties, saveAnimatedProperties, fps]) =>
            convert(port, disableEasing, saveKeyframes, saveForComparison, registry, comparisonProperties, saveAnimatedProperties, fps)
        """,
        [actual_port, disable_easing, save_keyframes, save_for_comparison, registry, comparison_properties,
         save_animated_properties, video_fps]
    )

    if disable_easing:
        print("Easing is disabled for all tweens.")


def convert_animation(html_file: str, port: int = 3013, create_video: bool = False, disable_easing: bool = False, save_keyframes: bool = False, save_for_comparison: bool = False, output_format: str = "mp4", video_fps: int = 30, print_console: bool = False, comparison_properties: dict | None = None, output_dir: str | None = None, save_animated_properties: bool = False, hide_grid: bool = False) -> None:
    """
    Convert a GSAP animation in an HTML file to JSON and optionally create animation output.
    
    Args:
        html_file (str): Path to the HTML file containing the GSAP animation
        port (int, optional): Port to run the server on. Use 0 to let the OS choose
            an available localhost port. Defaults to 3013.
        create_video (bool, optional): Whether to create animation output. Defaults to False.
        disable_easing (bool, optional): Set all GSAP tweens' easing to none. Defaults to False.
        save_keyframes (bool, optional): Whether to save keyframes data. Defaults to False.
        save_for_comparison (bool, optional): Whether to save rendered comparison data. Defaults to False.
        output_format (str, optional): Output format (mp4, gif, png, or svg). Defaults to "mp4".
            PNG and SVG formats write per-frame files to an output directory.
            GIF output requires FFmpeg.
        video_fps (int, optional): Frames per second for video, frame output, and JSON sampling. Defaults to 30.
        print_console (bool, optional): Whether to print console and network messages. Defaults to False.
        comparison_properties (dict, optional): Property config for comparison recording.
            Dict with keys 'spatial', 'visual', 'svgAttributes' mapping to lists of
            property names. None uses hardcoded defaults in convert.js.
        output_dir (str, optional): Directory to write output files to. None writes next to the HTML.
        save_animated_properties (bool, optional): Extract and save animated property names
            (registry names per element) to _properties.json. Defaults to False.
        hide_grid (bool, optional): Hide the SVG grid in raster captures without
            changing the interactive page styling. Defaults to False.
    """
    asyncio.run(run_conversion(html_file, port, create_video, disable_easing, save_keyframes, save_for_comparison, output_format, video_fps, print_console, comparison_properties, output_dir, save_animated_properties, hide_grid))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert JavaScript animation in an HTML file to JSON and optionally create animation output.")
    parser.add_argument("html_file", type=str, help="Path to the HTML file containing the JavaScript animation")
    parser.add_argument("port", type=int, help="Port to run the server on. Use 0 to let the OS choose an available localhost port")
    parser.add_argument("--create-video", "-v", action="store_true", help="Create animation output")
    parser.add_argument("--disable-easing", "-d", action="store_true", help="Set all GSAP tweens' easing to none")
    parser.add_argument("--save-keyframes", "-k", action="store_true", help="Save keyframes data to JSON")
    parser.add_argument("--save-for-comparison", "-c", action="store_true", help="Save rendered comparison data to JSON")
    parser.add_argument("--format", "-f", type=str, default="mp4", choices=["mp4", "gif", "png", "svg"], help="Output format for the animation: mp4, gif (requires FFmpeg), png, or svg frame sequence (default: mp4)")
    parser.add_argument("--video-fps", type=int, default=30, help="Frames per second for video, frame output, and JSON sampling (default: 30)")
    parser.add_argument("--print-console", "-pc", action="store_true", help="Print console and network messages from the browser (default: False)")
    parser.add_argument("--comparison-properties", type=str, default=None, help="JSON string of property config for comparison recording, e.g. '{\"spatial\": [\"transformedPts\", \"rotate\"], \"visual\": [\"opacity\"]}'")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write output files to (default: same directory as the HTML file)")
    parser.add_argument("--save-animated-properties", "-ap", action="store_true", help="Extract and save animated property names (registry names per element) to _properties.json")
    parser.add_argument("--hide-grid", action="store_true", help="Hide the SVG grid in raster captures (default: False)")
    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI usage."""
    args = parse_args()
    comp_props = json.loads(args.comparison_properties) if args.comparison_properties else None
    convert_animation(args.html_file, args.port, args.create_video, args.disable_easing, args.save_keyframes, args.save_for_comparison, args.format, args.video_fps, args.print_console, comp_props, args.output_dir, args.save_animated_properties, args.hide_grid)


if __name__ == "__main__":
    main()