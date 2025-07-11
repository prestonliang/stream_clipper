import os
import subprocess
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import tempfile

from config import config, get_quality_preset
from detector.detect import HypeMoment
from transcriber.transcribe import TranscriptSegment

logger = logging.getLogger(__name__)

class VideoClipper:
    """Handles video clipping, formatting, and subtitle generation"""
    
    def __init__(self):
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.assets_dir = Path(config.assets_dir)
        self.temp_dir = Path(config.temp_dir)
        
        self.quality_preset = get_quality_preset(config.video_quality)
    
    def create_clip(self, video_path: str, moment: HypeMoment, 
                   transcript_segments: List[TranscriptSegment] = None) -> Optional[str]:
        """
        Create a viral-ready clip from a hype moment
        
        Args:
            video_path: Path to source video
            moment: HypeMoment to clip
            transcript_segments: Transcript segments for subtitles
            
        Returns:
            Path to created clip or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clip_name = f"clip_{timestamp}_{moment.score:.1f}.mp4"
            output_path = self.output_dir / clip_name
            
            logger.info(f"Creating clip: {clip_name}")
            
            # Step 1: Extract and crop the video segment
            temp_clip = self._extract_segment(video_path, moment.start, moment.end)
            if not temp_clip:
                return None
            
            # Step 2: Convert to vertical format (9:16)
            temp_vertical = self._convert_to_vertical(temp_clip)
            if not temp_vertical:
                return None
            
            # Step 3: Add subtitles if transcript is available
            if transcript_segments:
                temp_with_subs = self._add_subtitles(temp_vertical, moment, transcript_segments)
                if temp_with_subs:
                    temp_vertical = temp_with_subs
            
            # Step 4: Add branding/effects
            temp_branded = self._add_branding(temp_vertical)
            if temp_branded:
                temp_vertical = temp_branded
            
            # Step 5: Final encoding with optimized settings
            if self._final_encode(temp_vertical, str(output_path)):
                logger.info(f"Successfully created clip: {output_path}")
                
                # Cleanup temp files
                self._cleanup_temp_files([temp_clip, temp_vertical, temp_branded])
                
                return str(output_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create clip: {e}")
            return None
    
    def _extract_segment(self, video_path: str, start: float, end: float) -> Optional[str]:
        """Extract video segment"""
        try:
            temp_file = self.temp_dir / f"segment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            duration = end - start
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start),
                '-t', str(duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-y',
                str(temp_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return str(temp_file)
            else:
                logger.error(f"Segment extraction failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract segment: {e}")
            return None
    
    def _convert_to_vertical(self, video_path: str) -> Optional[str]:
        """Convert video to 9:16 aspect ratio"""
        try:
            temp_file = self.temp_dir / f"vertical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            # Get video dimensions
            width, height = self._get_video_dimensions(video_path)
            if not width or not height:
                return None
            
            # Calculate crop parameters for 9:16 ratio
            target_aspect = 9 / 16
            current_aspect = width / height
            
            if current_aspect > target_aspect:
                # Video is too wide, crop sides
                new_width = int(height * target_aspect)
                crop_x = (width - new_width) // 2
                crop_filter = f"crop={new_width}:{height}:{crop_x}:0"
            else:
                # Video is too tall, crop top/bottom
                new_height = int(width / target_aspect)
                crop_y = (height - new_height) // 2
                crop_filter = f"crop={width}:{new_height}:0:{crop_y}"
            
            # Scale to target resolution
            target_scale = self.quality_preset['scale']
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', f"{crop_filter},scale={target_scale}",
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', self.quality_preset['audio_bitrate'],
                '-y',
                str(temp_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return str(temp_file)
            else:
                logger.error(f"Vertical conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to convert to vertical: {e}")
            return None
    
    def _get_video_dimensions(self, video_path: str) -> Tuple[Optional[int], Optional[int]]:
        """Get video width and height"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for stream in data['streams']:
                    if stream['codec_type'] == 'video':
                        return stream['width'], stream['height']
            
            return None, None
            
        except Exception as e:
            logger.error(f"Failed to get video dimensions: {e}")
            return None, None
    
    def _add_subtitles(self, video_path: str, moment: HypeMoment, 
                      transcript_segments: List[TranscriptSegment]) -> Optional[str]:
        """Add subtitles to video"""
        try:
            temp_file = self.temp_dir / f"subtitled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            # Find relevant transcript segments for this moment
            relevant_segments = []
            for segment in transcript_segments:
                if (segment.start >= moment.start and segment.end <= moment.end):
                    # Adjust timestamps relative to clip start
                    adjusted_segment = TranscriptSegment(
                        start=segment.start - moment.start,
                        end=segment.end - moment.start,
                        text=segment.text,
                        confidence=segment.confidence
                    )
                    relevant_segments.append(adjusted_segment)
            
            if not relevant_segments:
                return video_path  # No subtitles to add
            
            # Create SRT subtitle file
            srt_path = self._create_srt_file(relevant_segments)
            if not srt_path:
                return video_path
            
            # Add subtitles with styling
            subtitle_style = self._get_subtitle_style()
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', f"subtitles={srt_path}:force_style='{subtitle_style}'",
                '-c:a', 'copy',
                '-y',
                str(temp_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup SRT file
            Path(srt_path).unlink(missing_ok=True)
            
            if result.returncode == 0:
                return str(temp_file)
            else:
                logger.error(f"Subtitle addition failed: {result.stderr}")
                return video_path
                
        except Exception as e:
            logger.error(f"Failed to add subtitles: {e}")
            return video_path
    
    def _create_srt_file(self, segments: List[TranscriptSegment]) -> Optional[str]:
        """Create SRT subtitle file from transcript segments"""
        try:
            temp_srt = self.temp_dir / f"subtitles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
            
            with open(temp_srt, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._seconds_to_srt_time(segment.start)
                    end_time = self._seconds_to_srt_time(segment.end)
                    
                    # Split long text into multiple lines
                    text = self._format_subtitle_text(segment.text)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            
            return str(temp_srt)
            
        except Exception as e:
            logger.error(f"Failed to create SRT file: {e}")
            return None
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _format_subtitle_text(self, text: str, max_chars_per_line: int = 40) -> str:
        """Format subtitle text for better readability"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars_per_line:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines)
    
    def _get_subtitle_style(self) -> str:
        """Get subtitle styling parameters"""
        return (
            "FontName=Arial Bold,"
            "FontSize=24,"
            "PrimaryColour=&H00FFFFFF,"
            "SecondaryColour=&H00000000,"
            "OutlineColour=&H00000000,"
            "BackColour=&H80000000,"
            "Bold=1,"
            "Italic=0,"
            "Underline=0,"
            "StrikeOut=0,"
            "ScaleX=100,"
            "ScaleY=100,"
            "Spacing=0,"
            "Angle=0,"
            "BorderStyle=3,"
            "Outline=2,"
            "Shadow=0,"
            "Alignment=2,"
            "MarginL=10,"
            "MarginR=10,"
            "MarginV=20"
        )
    
    def _add_branding(self, video_path: str) -> Optional[str]:
        """Add branding elements to video"""
        try:
            # Check if branding assets exist
            logo_path = self.assets_dir / "logo.png"
            if not logo_path.exists():
                return video_path  # No branding to add
            
            temp_file = self.temp_dir / f"branded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            # Add logo overlay (top-right corner)
            cmd = [
                'ffmpeg', '-i', video_path,
                '-i', str(logo_path),
                '-filter_complex', 
                '[1:v]scale=120:120[logo];[0:v][logo]overlay=W-w-10:10',
                '-c:a', 'copy',
                '-y',
                str(temp_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return str(temp_file)
            else:
                logger.error(f"Branding addition failed: {result.stderr}")
                return video_path
                
        except Exception as e:
            logger.error(f"Failed to add branding: {e}")
            return video_path
    
    def _final_encode(self, video_path: str, output_path: str) -> bool:
        """Final encoding with optimized settings"""
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-maxrate', self.quality_preset['video_bitrate'],
                '-bufsize', str(int(self.quality_preset['video_bitrate'].replace('k', '')) * 2) + 'k',
                '-c:a', 'aac',
                '-b:a', self.quality_preset['audio_bitrate'],
                '-movflags', '+faststart',  # Optimize for streaming
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                logger.error(f"Final encoding failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed final encoding: {e}")
            return False
    
    def _cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        for file_path in file_paths:
            if file_path:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    def create_multiple_clips(self, video_path: str, moments: List[HypeMoment], 
                             transcript_segments: List[TranscriptSegment] = None) -> List[str]:
        """Create multiple clips from a list of moments"""
        created_clips = []
        
        for i, moment in enumerate(moments):
            logger.info(f"Creating clip {i+1}/{len(moments)}")
            clip_path = self.create_clip(video_path, moment, transcript_segments)
            
            if clip_path:
                created_clips.append(clip_path)
            else:
                logger.warning(f"Failed to create clip {i+1}")
        
        return created_clips
    
    def get_clip_info(self, clip_path: str) -> Dict:
        """Get information about a created clip"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams',
                clip_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                video_stream = None
                audio_stream = None
                
                for stream in data['streams']:
                    if stream['codec_type'] == 'video':
                        video_stream = stream
                    elif stream['codec_type'] == 'audio':
                        audio_stream = stream
                
                return {
                    'filename': Path(clip_path).name,
                    'filesize': int(data['format']['size']),
                    'duration': float(data['format']['duration']),
                    'width': video_stream['width'] if video_stream else None,
                    'height': video_stream['height'] if video_stream else None,
                    'video_bitrate': video_stream.get('bit_rate') if video_stream else None,
                    'audio_bitrate': audio_stream.get('bit_rate') if audio_stream else None,
                    'format': data['format']['format_name']
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get clip info: {e}")
            return {}

# Convenience functions
def create_clip_from_moment(video_path: str, moment: HypeMoment, 
                           transcript_segments: List[TranscriptSegment] = None) -> Optional[str]:
    """Create a single clip from a hype moment"""
    clipper = VideoClipper()
    return clipper.create_clip(video_path, moment, transcript_segments)

def create_all_clips(video_path: str, moments: List[HypeMoment], 
                    transcript_segments: List[TranscriptSegment] = None) -> List[str]:
    """Create clips for all detected moments"""
    clipper = VideoClipper()
    return clipper.create_multiple_clips(video_path, moments, transcript_segments)