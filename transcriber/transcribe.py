import os
import whisper
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from config import config

logger = logging.getLogger(__name__)

@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed audio"""
    start: float
    end: float
    text: str
    confidence: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start

class AudioTranscriber:
    """Handles audio transcription using OpenAI Whisper"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize transcriber with Whisper model
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_video(self, video_path: str, language: str = None) -> List[TranscriptSegment]:
        """
        Transcribe video file to text with timestamps
        
        Args:
            video_path: Path to video file
            language: Language code (e.g., "en", "es") or None for auto-detection
            
        Returns:
            List of TranscriptSegment objects
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded")
            
        try:
            logger.info(f"Transcribing video: {video_path}")
            
            # Whisper transcription options
            options = {
                "task": "transcribe",
                "language": language,
                "word_timestamps": True,
                "condition_on_previous_text": False,
                "temperature": 0.0,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            
            # Transcribe the video
            result = self.model.transcribe(video_path, **options)
            
            # Convert to our TranscriptSegment format
            segments = []
            for segment in result["segments"]:
                transcript_segment = TranscriptSegment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"].strip(),
                    confidence=segment.get("avg_logprob", 0.0)
                )
                segments.append(transcript_segment)
            
            logger.info(f"Transcribed {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []
    
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video file for transcription
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file or None if failed
        """
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                audio_path = tmp_audio.name
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',  # 16kHz sample rate (Whisper's preferred)
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return audio_path
            else:
                logger.error(f"Audio extraction failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return None
    
    def get_word_timestamps(self, video_path: str, language: str = None) -> List[Dict]:
        """
        Get word-level timestamps from video
        
        Args:
            video_path: Path to video file
            language: Language code or None for auto-detection
            
        Returns:
            List of word dictionaries with timestamps
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded")
            
        try:
            logger.info(f"Getting word timestamps for: {video_path}")
            
            result = self.model.transcribe(
                video_path, 
                language=language,
                word_timestamps=True,
                condition_on_previous_text=False
            )
            
            words = []
            for segment in result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        words.append({
                            "word": word["word"].strip(),
                            "start": word["start"],
                            "end": word["end"],
                            "confidence": word.get("probability", 0.0)
                        })
            
            return words
            
        except Exception as e:
            logger.error(f"Word timestamp extraction failed: {e}")
            return []
    
    def find_keyword_timestamps(self, segments: List[TranscriptSegment], keywords: List[str]) -> List[Dict]:
        """
        Find timestamps where specific keywords appear
        
        Args:
            segments: List of transcript segments
            keywords: List of keywords to search for
            
        Returns:
            List of keyword matches with timestamps
        """
        matches = []
        
        for segment in segments:
            text_lower = segment.text.lower()
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                if keyword_lower in text_lower:
                    matches.append({
                        "keyword": keyword,
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                        "confidence": segment.confidence
                    })
        
        return matches
    
    def get_transcript_summary(self, segments: List[TranscriptSegment]) -> Dict:
        """
        Get summary statistics of transcript
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Dictionary with summary statistics
        """
        if not segments:
            return {"total_duration": 0, "word_count": 0, "segment_count": 0}
        
        total_duration = max(seg.end for seg in segments)
        word_count = sum(len(seg.text.split()) for seg in segments)
        
        return {
            "total_duration": total_duration,
            "word_count": word_count,
            "segment_count": len(segments),
            "avg_words_per_minute": (word_count / total_duration) * 60 if total_duration > 0 else 0,
            "avg_segment_duration": total_duration / len(segments) if segments else 0
        }

# Convenience functions
def transcribe_video_file(video_path: str, model_size: str = "base") -> List[TranscriptSegment]:
    """Transcribe a video file"""
    transcriber = AudioTranscriber(model_size)
    return transcriber.transcribe_video(video_path)

def find_hype_moments(video_path: str, keywords: List[str] = None) -> List[Dict]:
    """Find hype moments in video based on keywords"""
    transcriber = AudioTranscriber("base")
    segments = transcriber.transcribe_video(video_path)
    
    if keywords is None:
        keywords = config.hype_keywords
    
    return transcriber.find_keyword_timestamps(segments, keywords)

def get_video_transcript_summary(video_path: str) -> Dict:
    """Get transcript summary for video"""
    transcriber = AudioTranscriber("base")
    segments = transcriber.transcribe_video(video_path)
    return transcriber.get_transcript_summary(segments)