import os
import subprocess
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pathlib import Path
import yt_dlp

from config import config

logger = logging.getLogger(__name__)

class StreamDownloader:
    """Downloads stream segments from Twitch or YouTube"""
    
    def __init__(self):
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # yt-dlp options for live stream downloading
        self.ydl_opts = {
            'format': 'best[height<=1080]',
            'outtmpl': str(self.temp_dir / 'segment_%(timestamp)s.%(ext)s'),
            'writesubtitles': False,
            'writeautomaticsub': False,
            'no_warnings': True,
            'extractaudio': False,
            'audioformat': 'mp3',
            'audioquality': '192',
        }
    
    def download_segment(self, stream_url: str, duration: int = None) -> Optional[str]:
        """
        Download a segment from live stream
        
        Args:
            stream_url: URL of the stream
            duration: Duration in seconds (defaults to config.segment_duration)
            
        Returns:
            Path to downloaded segment file or None if failed
        """
        if duration is None:
            duration = config.segment_duration
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.temp_dir / f"segment_{timestamp}.mp4"
        
        try:
            # For live streams, we'll use yt-dlp with time limits
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['outtmpl'] = str(output_path.with_suffix('.%(ext)s'))
            
            # Add live stream options
            if self._is_live_stream(stream_url):
                ydl_opts.update({
                    'live_from_start': False,
                    'wait_for_video': (1, 5),  # Wait 1-5 seconds for video
                    'fragment_retries': 3,
                    'hls_use_mpegts': True,
                })
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading segment from {stream_url}")
                ydl.download([stream_url])
                
            # Find the actual downloaded file
            downloaded_files = list(self.temp_dir.glob(f"segment_{timestamp}.*"))
            if downloaded_files:
                actual_file = downloaded_files[0]
                
                # If we need to trim to specific duration, use ffmpeg
                if duration and duration < 600:  # Only trim if less than 10 minutes
                    trimmed_file = self._trim_segment(actual_file, duration)
                    if trimmed_file:
                        actual_file.unlink()  # Remove original
                        return str(trimmed_file)
                
                return str(actual_file)
            else:
                logger.error("No files downloaded")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download segment: {e}")
            return None
    
    def _is_live_stream(self, url: str) -> bool:
        """Check if URL is a live stream"""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('is_live', False)
        except:
            return False
    
    def _trim_segment(self, input_file: Path, duration: int) -> Optional[Path]:
        """Trim segment to specified duration using ffmpeg"""
        output_file = input_file.with_name(f"trimmed_{input_file.name}")
        
        try:
            cmd = [
                'ffmpeg', '-i', str(input_file),
                '-t', str(duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-y',  # Overwrite output file
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return output_file
            else:
                logger.error(f"FFmpeg trim failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to trim segment: {e}")
            return None
    
    def download_vod_segment(self, video_url: str, start_time: str, duration: int) -> Optional[str]:
        """
        Download specific segment from VOD
        
        Args:
            video_url: URL of the VOD
            start_time: Start time in format "HH:MM:SS" or seconds
            duration: Duration in seconds
            
        Returns:
            Path to downloaded segment
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.temp_dir / f"vod_segment_{timestamp}.mp4"
        
        try:
            # Convert start_time to seconds if it's in HH:MM:SS format
            if isinstance(start_time, str) and ':' in start_time:
                time_parts = start_time.split(':')
                start_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
            else:
                start_seconds = int(start_time)
            
            cmd = [
                'ffmpeg',
                '-ss', str(start_seconds),
                '-i', video_url,
                '-t', str(duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return str(output_path)
            else:
                logger.error(f"VOD download failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download VOD segment: {e}")
            return None
    
    def get_stream_info(self, stream_url: str) -> Optional[dict]:
        """Get stream information"""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(stream_url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'is_live': info.get('is_live', False),
                    'duration': info.get('duration'),
                    'view_count': info.get('view_count', 0),
                    'formats': len(info.get('formats', [])),
                    'url': stream_url
                }
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return None
    
    def cleanup_old_segments(self, max_age_hours: int = 24):
        """Remove old segment files"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for file_path in self.temp_dir.glob("segment_*"):
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    file_path.unlink()
                    logger.info(f"Cleaned up old segment: {file_path.name}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old segments: {e}")

# Convenience functions
def download_latest_segment(stream_url: str = None, duration: int = None) -> Optional[str]:
    """Download latest segment from configured stream"""
    downloader = StreamDownloader()
    url = stream_url or config.stream_url
    return downloader.download_segment(url, duration)

def get_stream_info(stream_url: str = None) -> Optional[dict]:
    """Get stream information"""
    downloader = StreamDownloader()
    url = stream_url or config.stream_url
    return downloader.get_stream_info(url)