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


def create_video_from_frames(frames: List[np.ndarray], output_path: str, fps: int = 60) -> None:
    """Create a video from a list of frames using OpenCV."""
    if not frames:
        raise ValueError("No frames provided")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        for frame in frames:
            out.write(frame)
    finally:
        out.release()

    ## Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        # Use ffmpeg to make the video web-compatible
        output_path = Path(output_path)
        temp_path = output_path.with_suffix('.temp.mp4')
        output_path.rename(temp_path)
        subprocess.run([
            'ffmpeg', '-y',
            '-i', str(temp_path),
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(output_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_path.unlink()
    except (subprocess.SubprocessError, FileNotFoundError):
        print("ffmpeg not found, skipping web optimization step")


def setup_fastapi_app(html_file: str, html_dir: str, base_name: str) -> FastAPI:
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

    @app.post("/create-video")
    async def create_video(request: Request):
        """Create a video from SVG frames."""
        data = await request.json()
        svg_frames = data['frames']
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

            video_path = Path(html_dir) / f"{base_name}_animation.mp4"
            create_video_from_frames(frames, str(video_path))
            
            print(f"MoVer converter: Animation saved as video to {video_path}")
            return JSONResponse(content={"success": True, "path": str(video_path)})
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

    @app.get("/")
    async def serve_html():
        """Serve the HTML file."""
        return FileResponse(html_file)

    # Mount the assets directory at root to allow relative path access
    assets_path = Path(__file__).parent / "assets"
    app.mount("/", StaticFiles(directory=str(assets_path)), name="assets")

    return app


async def run_conversion(html_file: str, port: int, create_video: bool = False) -> None:
    """Run the conversion process."""
    html_path = Path(html_file)
    html_dir = str(html_path.parent)
    base_name = html_path.stem
    
    app = setup_fastapi_app(html_file, html_dir, base_name)

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
            await page.evaluate(f"convert({port})")

            if create_video:
                print("Creating video...")
                await page.evaluate(f"createVideo({port})")

            await browser.close()
            
    finally:
        # Stop the server
        server.should_exit = True
        await server.shutdown()


def convert_animation(html_file: str, port: int = 3013, create_video: bool = False) -> None:
    """
    Convert a GSAP animation in an HTML file to JSON and optionally create a video.
    
    Args:
        html_file (str): Path to the HTML file containing the GSAP animation
        port (int, optional): Port to run the server on. Defaults to 3013.
        create_video (bool, optional): Whether to create a video. Defaults to False.
    """
    asyncio.run(run_conversion(html_file, port, create_video))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert JavaScript animation in an HTML file to JSON and optionally create a video.")
    parser.add_argument("html_file", type=str, help="Path to the HTML file containing the JavaScript animation")
    parser.add_argument("port", type=int, help="Port to run the server on")
    parser.add_argument("--create-video", "-v", action="store_true", help="Create a video of the animation")
    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI usage."""
    args = parse_args()
    convert_animation(args.html_file, args.port, args.create_video)


if __name__ == "__main__":
    main()