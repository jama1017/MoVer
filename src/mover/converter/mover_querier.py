"""
MoVer Querier - Experimental SVGPT utility for querying animation positions.

This module provides functionality to find the time(s) when an element is at 
specific positions during animation in HTML files containing GSAP animations.
"""

import json
import asyncio
import argparse
from typing import List
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright
import uvicorn

from mover.converter.mover_converter import _get_bound_port, _wait_for_server_start


def setup_fastapi_app(html_file: str, html_dir: str, base_name: str) -> FastAPI:
    """Set up and configure a minimal FastAPI application for position querying."""
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def serve_html():
        """Serve the HTML file."""
        return FileResponse(html_file)

    # Mount the assets directory at root to allow relative path access
    assets_path = Path(__file__).parent / "assets"
    app.mount("/", StaticFiles(directory=str(assets_path)), name="assets")

    return app


async def run_get_position(html_file: str, target_centroids: List[dict], element_id: str, tolerance: float, port: int) -> List[List[dict]]:
    """
    Run the time analysis.
    """
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
        await _wait_for_server_start(server, server_task)
        actual_port = _get_bound_port(server)

        # Initialize Playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page()
                await page.goto(
                    f"http://127.0.0.1:{actual_port}",
                    wait_until="networkidle",
                )
                await page.evaluate("document.fonts.ready")

                # Playwright serializes values safely into the page context.
                return await page.evaluate(
                    """({targetCentroids, elementId, tolerance}) =>
                        getPositionInTime(targetCentroids, elementId, tolerance)
                    """,
                    {
                        "targetCentroids": target_centroids,
                        "elementId": element_id,
                        "tolerance": tolerance,
                    },
                )
            finally:
                await browser.close()
            
    finally:
        # Stop the server
        server.should_exit = True
        if not server_task.done():
            await server_task


def get_position_in_time(html_file: str, target_centroids: List[dict], element_id: str, tolerance: float = 0.1, port: int = 3014) -> List[List[dict]]:
    """
    Find the time(s) when an element is at specific positions during animation.
    
    Args:
        html_file (str): Path to the HTML file containing the GSAP animation
        target_centroids (List[dict]): List of target positions with 'x' and 'y' keys (e.g., [{'x': 100, 'y': 200}, {'x': 300, 'y': 150}])
        element_id (str): ID of the element to track
        tolerance (float, optional): Tolerance for position matching in pixels. Defaults to 0.1.
        port (int, optional): Port to run the server on. Defaults to 3014.
    
    Returns:
        List[List[dict]]: List of match lists, one for each centroid, with 'time' and 'error' keys
    """
    return asyncio.run(run_get_position(html_file, target_centroids, element_id, tolerance, port))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Query element positions during GSAP animation in an HTML file.")
    parser.add_argument("html_file", type=str, help="Path to the HTML file containing the GSAP animation")
    parser.add_argument("port", type=int, help="Port to run the server on; use 0 for an available port")
    parser.add_argument("target_centroids", type=str, help="Target centroids as 'x1,y1;x2,y2;...'")
    parser.add_argument("element_id", type=str, help="ID of the element to track")
    parser.add_argument("tolerance", type=float, help="Tolerance for position matching in pixels")
    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI usage."""
    args = parse_args()
    
    # Parse multiple centroids separated by semicolons
    target_centroids = []
    for centroid_str in args.target_centroids.split(';'):
        x_str, y_str = centroid_str.split(',')
        target_centroids.append({'x': float(x_str), 'y': float(y_str)})
    
    matches = get_position_in_time(args.html_file, target_centroids, args.element_id, args.tolerance, args.port)
    print(json.dumps(matches, indent=2))


if __name__ == "__main__":
    main()