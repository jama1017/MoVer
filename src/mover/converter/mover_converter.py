import os
import json
import asyncio
import argparse
from typing import List
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright
import subprocess
from PIL import Image
import io
import cairosvg
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

    @app.post("/create-video")
    async def create_video(request: Request):
        """Create a video from SVG frames."""
        data = await request.json()
        svg_frames = data['frames']
        fps = data.get('fps', 30)  ## Get fps from request, default to 30
        frames = []

        try:
            for svg_data in svg_frames:
                # Convert SVG to PNG using cairosvg
                png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
                
                # Open PNG with PIL and ensure white background
                img = Image.open(io.BytesIO(png_data))
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                
                # Convert to numpy array for OpenCV
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                frames.append(frame)

            output_path = Path(html_dir) / f"{base_name}_animation.{output_format}"
            create_video_from_frames(frames, str(output_path), fps=fps, output_format=output_format)
            
            print(f"MoVer converter: Animation saved as {output_format.upper()} to {output_path}")
            return JSONResponse(content={"success": True, "path": str(output_path)})
        
        except Exception as e:
            print(f"Error creating {output_format}: {str(e)}")
            return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

    @app.get("/")
    async def serve_html():
        """Serve the HTML file."""
        return FileResponse(html_file)

    # Mount the assets directory at root to allow relative path access
    assets_path = Path(__file__).parent / "assets"
    app.mount("/", StaticFiles(directory=str(assets_path)), name="assets")

    return app


async def run_conversion(html_file: str, port: int, create_video: bool = False, disable_easing: bool = False, save_keyframes: bool = False, output_format: str = "mp4", video_fps: int = 30) -> None:
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

            print("Page loading time: ", end="", flush=True)
            start_time = asyncio.get_event_loop().time()
            await page.goto(f"http://127.0.0.1:{port}")
            load_time = asyncio.get_event_loop().time() - start_time
            print(f"{load_time:.2f} seconds")

            # Execute JavaScript in the page context
            await page.evaluate(f"convert({port}, {str(disable_easing).lower()}, {str(save_keyframes).lower()})")
            
            if disable_easing:
                print("Easing is disabled for all tweens.")

            if create_video:
                print(f"Creating {output_format.upper()}...")
                await page.evaluate(f"createVideo({port}, {video_fps})")

            await browser.close()
            
    finally:
        # Stop the server
        server.should_exit = True
        await server.shutdown()


def convert_animation(html_file: str, port: int = 3013, create_video: bool = False, disable_easing: bool = False, save_keyframes: bool = False, output_format: str = "mp4", video_fps: int = 30) -> None:
    """
    Convert a GSAP animation in an HTML file to JSON and optionally create a video.
    
    Args:
        html_file (str): Path to the HTML file containing the GSAP animation
        port (int, optional): Port to run the server on. Defaults to 3013.
        create_video (bool, optional): Whether to create a video. Defaults to False.
        disable_easing (bool, optional): Set all GSAP tweens' easing to none. Defaults to False.
        save_keyframes (bool, optional): Whether to save keyframes data. Defaults to False.
        output_format (str, optional): Output format (mp4 or gif). Defaults to "mp4".
        video_fps (int, optional): Frames per second for video creation. Defaults to 12.
    """
    asyncio.run(run_conversion(html_file, port, create_video, disable_easing, save_keyframes, output_format, video_fps))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert JavaScript animation in an HTML file to JSON and optionally create a video.")
    parser.add_argument("html_file", type=str, help="Path to the HTML file containing the JavaScript animation")
    parser.add_argument("port", type=int, help="Port to run the server on")
    parser.add_argument("--create-video", "-v", action="store_true", help="Create a video of the animation")
    parser.add_argument("--disable-easing", "-d", action="store_true", help="Set all GSAP tweens' easing to none")
    parser.add_argument("--save-keyframes", "-k", action="store_true", help="Save keyframes data to JSON")
    parser.add_argument("--format", "-f", type=str, default="mp4", choices=["mp4", "gif"], help="Output format for the animation (default: mp4)")
    parser.add_argument("--video-fps", type=int, default=30, help="Frames per second for video creation (default: 30)")
    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI usage."""
    args = parse_args()
    convert_animation(args.html_file, args.port, args.create_video, args.disable_easing, args.save_keyframes, args.format, args.video_fps)


if __name__ == "__main__":
    main()