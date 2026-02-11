import os
import json
import asyncio
import argparse
import tempfile
import shutil
from typing import List, Optional
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
import io
import uvicorn


def create_video_from_frames(frames: List[np.ndarray], output_path: str, fps: int = 30, output_format: str = "mp4") -> None:
    """Create a video or GIF from a list of frames using OpenCV and ffmpeg."""
    if not frames:
        raise ValueError("No frames provided")
    
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

    ## Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        
        if output_format == "gif":
            ## Convert MP4 to GIF with optimized palette
            subprocess.run([
                'ffmpeg', '-y',
                '-i', str(temp_mp4_path),
                '-vf', f'fps={fps},split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5',
                str(output_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            temp_mp4_path.unlink()
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
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            temp_mp4_path.unlink()
    except (subprocess.SubprocessError, FileNotFoundError):
        print("ffmpeg not found, skipping conversion step")
        ## If ffmpeg fails, just rename the temp file
        if temp_mp4_path.exists():
            temp_mp4_path.rename(output_path)


async def capture_frames_server_driven(
    page: Page,
    output_path: str,
    fps: int = 30,
    output_format: str = "mp4",
) -> None:
    """
    Server-driven frame capture: Python controls the timeline and screenshots.
    Frames are streamed to disk to avoid holding all in memory.
    """
    # 1. Get animation info from the page
    anim_info = await page.evaluate("() => getAnimationInfo()")
    duration = anim_info["animDuration"]
    total_frames = anim_info["steps"]

    print(f"Capturing {total_frames + 1} frames at {fps} FPS (duration: {duration}s)")

    # 2. Hide GSDevTools if present (it overlays on top of the SVG)
    await page.evaluate("""() => {
        const devtools = document.querySelector('#GSDevTools');
        if (devtools) devtools.style.display = 'none';
        // Also hide any GSDevTools container elements
        document.querySelectorAll('[class*="gs-dev-tools"]').forEach(el => el.style.display = 'none');
    }""")

    # 3. Locate the SVG element once
    svg_element = page.locator("svg").first
    await svg_element.wait_for(state="visible")

    # 3. Stream frames to a temp directory
    temp_dir = tempfile.mkdtemp(prefix="mover_frames_")

    try:
        for frame_index in range(total_frames + 1):
            # Seek the timeline from Python using the existing JS helper
            await page.evaluate(f"() => seekToFrame({frame_index}, {fps}, {duration})")

            # Wait for repaint
            await page.evaluate(
                "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
            )

            # Screenshot the SVG element directly to disk
            frame_path = Path(temp_dir) / f"frame_{frame_index:06d}.png"
            await svg_element.screenshot(path=str(frame_path), type="png")

            # Progress reporting
            if (frame_index + 1) % max(1, (total_frames + 1) // 10) == 0 or frame_index == total_frames:
                print(f"  Captured frame {frame_index + 1}/{total_frames + 1}")

        # 4. Encode video using FFmpeg directly from image sequence
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)

            if output_format == "gif":
                subprocess.run([
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', str(Path(temp_dir) / 'frame_%06d.png'),
                    '-vf', f'fps={fps},split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5',
                    str(output_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                subprocess.run([
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', str(Path(temp_dir) / 'frame_%06d.png'),
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    str(output_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            print(f"Video saved to {output_path}")

        except (subprocess.SubprocessError, FileNotFoundError):
            print("ffmpeg not found, falling back to OpenCV encoding")
            # Fallback: read frames back and use OpenCV
            frames = []
            for frame_index in range(total_frames + 1):
                frame_path = Path(temp_dir) / f"frame_{frame_index:06d}.png"
                img = Image.open(frame_path)
                if img.mode != 'RGB':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[3])
                    else:
                        background.paste(img)
                    img = background
                frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            create_video_from_frames(frames, output_path, fps, output_format)
            print(f"Video saved to {output_path} (OpenCV fallback)")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def setup_fastapi_app(html_file: str, html_dir: str, base_name: str, output_format: str = "mp4") -> FastAPI:
    """Set up and configure the FastAPI application."""
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
        json_file_path = Path(html_dir) / f"{base_name}_data.json"
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print("SAVED TO LOCAL")
        return JSONResponse(content={"status": "success"})
    
    @app.post("/convert-js-to-keyframes-json")
    async def convert_js_to_keyframes_json(request: Request):
        """Convert JavaScript keyframes data to JSON and save it."""
        json_data = await request.json()
        json_file_path = Path(html_dir) / f"{base_name}_data_keyframes.json"
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print("SAVED KEYFRAMES TO LOCAL")
        return JSONResponse(content={"status": "success"})

    @app.post("/convert-js-to-rendered-json")
    async def convert_js_to_rendered_json(request: Request):
        """Convert JavaScript rendered comparison data to JSON and save it."""
        json_data = await request.json()
        json_file_path = Path(html_dir) / f"{base_name}_data_rendered.json"
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print("SAVED RENDERED DATA TO LOCAL")
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


async def run_conversion(html_file: str, port: int, create_video: bool = False, disable_easing: bool = False, save_keyframes: bool = False, save_for_comparison: bool = False, output_format: str = "mp4", video_fps: int = 30, print_console: bool = False) -> None:
    """Run the conversion process."""
    html_path = Path(html_file)
    html_dir = str(html_path.parent)
    base_name = html_path.stem
    
    app = setup_fastapi_app(html_file, html_dir, base_name, output_format)

    # Configure uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    
    # Start the server as a task
    server_task = asyncio.create_task(server.serve())

    try:
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
            await page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")
            ## Explicitly wait for all fonts to be ready
            await page.evaluate("document.fonts.ready")
            print("Fonts loaded")
            load_time = asyncio.get_event_loop().time() - start_time
            print(f"{load_time:.2f} seconds")

            # Execute JavaScript in the page context
            await page.evaluate(f"convert({port}, {str(disable_easing).lower()}, {str(save_keyframes).lower()}, {str(save_for_comparison).lower()})")
            
            if disable_easing:
                print("Easing is disabled for all tweens.")

            # Server-driven video creation — no HTTP round-trips per frame
            if create_video:
                print(f"Creating {output_format.upper()}...")
                output_path = str(Path(html_dir) / f"{base_name}_animation.{output_format}")
                await capture_frames_server_driven(page, output_path, video_fps, output_format)

            await browser.close()
            
    finally:
        # Stop the server
        server.should_exit = True
        await server.shutdown()


def convert_animation(html_file: str, port: int = 3013, create_video: bool = False, disable_easing: bool = False, save_keyframes: bool = False, save_for_comparison: bool = False, output_format: str = "mp4", video_fps: int = 30, print_console: bool = False) -> None:
    """
    Convert a GSAP animation in an HTML file to JSON and optionally create a video.
    
    Args:
        html_file (str): Path to the HTML file containing the GSAP animation
        port (int, optional): Port to run the server on. Defaults to 3013.
        create_video (bool, optional): Whether to create a video. Defaults to False.
        disable_easing (bool, optional): Set all GSAP tweens' easing to none. Defaults to False.
        save_keyframes (bool, optional): Whether to save keyframes data. Defaults to False.
        save_for_comparison (bool, optional): Whether to save rendered comparison data. Defaults to False.
        output_format (str, optional): Output format (mp4 or gif). Defaults to "mp4".
        video_fps (int, optional): Frames per second for video creation. Defaults to 30.
        print_console (bool, optional): Whether to print console and network messages. Defaults to False.
    """
    asyncio.run(run_conversion(html_file, port, create_video, disable_easing, save_keyframes, save_for_comparison, output_format, video_fps, print_console))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert JavaScript animation in an HTML file to JSON and optionally create a video.")
    parser.add_argument("html_file", type=str, help="Path to the HTML file containing the JavaScript animation")
    parser.add_argument("port", type=int, help="Port to run the server on")
    parser.add_argument("--create-video", "-v", action="store_true", help="Create a video of the animation")
    parser.add_argument("--disable-easing", "-d", action="store_true", help="Set all GSAP tweens' easing to none")
    parser.add_argument("--save-keyframes", "-k", action="store_true", help="Save keyframes data to JSON")
    parser.add_argument("--save-for-comparison", "-c", action="store_true", help="Save rendered comparison data to JSON")
    parser.add_argument("--format", "-f", type=str, default="mp4", choices=["mp4", "gif"], help="Output format for the animation (default: mp4)")
    parser.add_argument("--video-fps", type=int, default=30, help="Frames per second for video creation (default: 30)")
    parser.add_argument("--print-console", "-pc", action="store_true", help="Print console and network messages from the browser (default: False)")
    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI usage."""
    args = parse_args()
    convert_animation(args.html_file, args.port, args.create_video, args.disable_easing, args.save_keyframes, args.save_for_comparison, args.format, args.video_fps, args.print_console)


if __name__ == "__main__":
    main()