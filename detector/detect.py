import os
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import subprocess
import json
import tempfile
from pathlib import Path

from config import config
from transcriber.transcribe import TranscriptSegment

logger = logging.getLogger(__name__)

@dataclass
class HypeMoment:
    """Represents a detected hype moment"""
    start: float
    end: float
    score: float
    triggers: List[str]  # What triggered this moment (keywords, volume, etc.)
    transcript: str
    confidence: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start

class HypeDetector:
    """Detects hype moments in video based on audio and transcript analysis"""
    
    def __init__(self):
        self.volume_threshold = config.volume_threshold
        self.keyword_boost = config.keyword_boost
        self.hype_keywords = [kw.lower() for kw in config.hype_keywords]
    
    def analyze_video(self, video_path: str, transcript_segments: List[TranscriptSegment]) -> List[HypeMoment]:
        """
        Analyze video for hype moments using both audio and transcript
        
        Args:
            video_path: Path to video file
            transcript_segments: List of transcript segments
            
        Returns:
            List of detected hype moments
        """
        try:
            logger.info(f"Analyzing video for hype moments: {video_path}")
            
            # Get audio analysis
            audio_peaks = self._analyze_audio_volume(video_path)
            
            # Get keyword matches
            keyword_matches = self._find_keyword_matches(transcript_segments)
            
            # Get speech rate analysis
            speech_rate_spikes = self._analyze_speech_rate(transcript_segments)
            
            # Combine all signals to detect hype moments
            hype_moments = self._combine_signals(
                audio_peaks, 
                keyword_matches, 
                speech_rate_spikes,
                transcript_segments
            )
            
            # Filter and rank moments
            filtered_moments = self._filter_and_rank_moments(hype_moments)
            
            logger.info(f"Detected {len(filtered_moments)} hype moments")
            return filtered_moments
            
        except Exception as e:
            logger.error(f"Hype detection failed: {e}")
            return []
    
    def _analyze_audio_volume(self, video_path: str) -> List[Dict]:
        """Analyze audio volume to find spikes"""
        try:
            # Use ffmpeg to analyze audio volume
            cmd = [
                'ffmpeg', '-i', video_path,
                '-af', 'volumedetect,astats=metadata=1:reset=1',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse volume information from stderr
            volume_peaks = []
            lines = result.stderr.split('\n')
            
            current_time = 0
            for line in lines:
                if 'lavfi.astats.Overall.RMS_level' in line:
                    try:
                        # Extract RMS level
                        rms_level = float(line.split('=')[1].strip())
                        
                        # Convert to 0-1 scale (rough approximation)
                        volume_score = max(0, min(1, (rms_level + 60) / 60))
                        
                        if volume_score > self.volume_threshold:
                            volume_peaks.append({
                                'time': current_time,
                                'volume': volume_score,
                                'type': 'volume_spike'
                            })
                        
                        current_time += 1  # Increment time
                    except (ValueError, IndexError):
                        continue
            
            return volume_peaks
            
        except Exception as e:
            logger.error(f"Audio volume analysis failed: {e}")
            return []
    
    def _find_keyword_matches(self, segments: List[TranscriptSegment]) -> List[Dict]:
        """Find segments containing hype keywords"""
        matches = []
        
        for segment in segments:
            text_lower = segment.text.lower()
            found_keywords = []
            
            for keyword in self.hype_keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                # Calculate keyword score based on number and type of keywords
                keyword_score = len(found_keywords) * self.keyword_boost
                
                matches.append({
                    'start': segment.start,
                    'end': segment.end,
                    'score': keyword_score,
                    'keywords': found_keywords,
                    'text': segment.text,
                    'type': 'keyword_match'
                })
        
        return matches
    
    def _analyze_speech_rate(self, segments: List[TranscriptSegment]) -> List[Dict]:
        """Analyze speech rate to find excitement spikes"""
        speech_spikes = []
        
        # Calculate average speech rate
        total_words = sum(len(seg.text.split()) for seg in segments)
        total_duration = sum(seg.duration for seg in segments)
        avg_wpm = (total_words / total_duration) * 60 if total_duration > 0 else 0
        
        for segment in segments:
            words = len(segment.text.split())
            duration = segment.duration
            
            if duration > 0:
                segment_wpm = (words / duration) * 60
                
                # If speech rate is significantly higher than average
                if segment_wpm > avg_wpm * 1.5:
                    speech_spikes.append({
                        'start': segment.start,
                        'end': segment.end,
                        'wpm': segment_wpm,
                        'avg_wpm': avg_wpm,
                        'score': segment_wpm / avg_wpm,
                        'text': segment.text,
                        'type': 'speech_spike'
                    })
        
        return speech_spikes
    
    def _combine_signals(self, audio_peaks: List[Dict], keyword_matches: List[Dict], 
                        speech_spikes: List[Dict], segments: List[TranscriptSegment]) -> List[HypeMoment]:
        """Combine all signals to detect hype moments"""
        moments = []
        
        # Create time-based grid for combining signals
        if not segments:
            return moments
            
        max_time = max(seg.end for seg in segments)
        time_grid = np.arange(0, max_time, 0.5)  # 0.5 second intervals
        
        # Score each time interval
        for t in time_grid:
            score = 0
            triggers = []
            relevant_text = ""
            
            # Check for audio peaks
            for peak in audio_peaks:
                if abs(peak['time'] - t) < 2:  # Within 2 seconds
                    score += peak['volume'] * 2
                    triggers.append(f"volume({peak['volume']:.2f})")
            
            # Check for keyword matches
            for match in keyword_matches:
                if match['start'] <= t <= match['end']:
                    score += match['score']
                    triggers.extend([f"keyword({kw})" for kw in match['keywords']])
                    relevant_text = match['text']
            
            # Check for speech spikes
            for spike in speech_spikes:
                if spike['start'] <= t <= spike['end']:
                    score += spike['score']
                    triggers.append(f"speech({spike['wpm']:.0f}wpm)")
                    if not relevant_text:
                        relevant_text = spike['text']
            
            # If score is high enough, create a hype moment
            if score >= 2.0:  # Threshold for hype moment
                # Find the best transcript for this moment
                if not relevant_text:
                    for segment in segments:
                        if segment.start <= t <= segment.end:
                            relevant_text = segment.text
                            break
                
                moments.append(HypeMoment(
                    start=max(0, t - 2),  # Start 2 seconds before
                    end=min(max_time, t + 3),  # End 3 seconds after
                    score=score,
                    triggers=triggers,
                    transcript=relevant_text,
                    confidence=min(1.0, score / 5.0)  # Normalize confidence
                ))
        
        return moments
    
    def _filter_and_rank_moments(self, moments: List[HypeMoment]) -> List[HypeMoment]:
        """Filter overlapping moments and rank by score"""
        if not moments:
            return []
        
        # Sort by score (descending)
        sorted_moments = sorted(moments, key=lambda m: m.score, reverse=True)
        
        # Remove overlapping moments (keep highest scoring)
        filtered_moments = []
        for moment in sorted_moments:
            # Check if this moment overlaps with any already selected
            overlaps = False
            for existing in filtered_moments:
                if (moment.start < existing.end and moment.end > existing.start):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_moments.append(moment)
        
        # Ensure moments are within duration limits
        valid_moments = []
        for moment in filtered_moments:
            duration = moment.duration
            if config.min_clip_duration <= duration <= config.max_clip_duration:
                valid_moments.append(moment)
            elif duration > config.max_clip_duration:
                # Trim to max duration
                trimmed_moment = HypeMoment(
                    start=moment.start,
                    end=moment.start + config.max_clip_duration,
                    score=moment.score,
                    triggers=moment.triggers,
                    transcript=moment.transcript,
                    confidence=moment.confidence
                )
                valid_moments.append(trimmed_moment)
        
        return valid_moments
    
    def detect_silence_breaks(self, video_path: str) -> List[Tuple[float, float]]:
        """Detect silence breaks in audio for better clip boundaries"""
        try:
            # Use ffmpeg to detect silence
            cmd = [
                'ffmpeg', '-i', video_path,
                '-af', 'silencedetect=noise=-30dB:duration=0.5',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            silence_breaks = []
            lines = result.stderr.split('\n')
            
            for line in lines:
                if 'silence_start' in line:
                    try:
                        start_time = float(line.split('silence_start: ')[1].split()[0])
                        silence_breaks.append((start_time, None))
                    except (ValueError, IndexError):
                        continue
                elif 'silence_end' in line and silence_breaks:
                    try:
                        end_time = float(line.split('silence_end: ')[1].split()[0])
                        if silence_breaks[-1][1] is None:
                            silence_breaks[-1] = (silence_breaks[-1][0], end_time)
                    except (ValueError, IndexError):
                        continue
            
            return [(start, end) for start, end in silence_breaks if end is not None]
            
        except Exception as e:
            logger.error(f"Silence detection failed: {e}")
            return []
    
    def refine_clip_boundaries(self, moment: HypeMoment, video_path: str) -> HypeMoment:
        """Refine clip boundaries to avoid cutting mid-sentence"""
        try:
            silence_breaks = self.detect_silence_breaks(video_path)
            
            if not silence_breaks:
                return moment
            
            # Find the best start boundary
            best_start = moment.start
            for silence_start, silence_end in silence_breaks:
                if abs(silence_end - moment.start) < 2:  # Within 2 seconds
                    best_start = silence_end
                    break
            
            # Find the best end boundary
            best_end = moment.end
            for silence_start, silence_end in silence_breaks:
                if abs(silence_start - moment.end) < 2:  # Within 2 seconds
                    best_end = silence_start
                    break
            
            return HypeMoment(
                start=best_start,
                end=best_end,
                score=moment.score,
                triggers=moment.triggers,
                transcript=moment.transcript,
                confidence=moment.confidence
            )
            
        except Exception as e:
            logger.error(f"Boundary refinement failed: {e}")
            return moment
    
    def get_detection_stats(self, moments: List[HypeMoment]) -> Dict:
        """Get statistics about detected moments"""
        if not moments:
            return {"total_moments": 0, "avg_score": 0, "total_duration": 0}
        
        total_duration = sum(moment.duration for moment in moments)
        avg_score = sum(moment.score for moment in moments) / len(moments)
        
        trigger_counts = {}
        for moment in moments:
            for trigger in moment.triggers:
                trigger_type = trigger.split('(')[0]
                trigger_counts[trigger_type] = trigger_counts.get(trigger_type, 0) + 1
        
        return {
            "total_moments": len(moments),
            "avg_score": avg_score,
            "total_duration": total_duration,
            "avg_duration": total_duration / len(moments),
            "trigger_counts": trigger_counts,
            "top_triggers": sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

# Convenience functions
def detect_hype_moments(video_path: str, transcript_segments: List[TranscriptSegment]) -> List[HypeMoment]:
    """Detect hype moments in video"""
    detector = HypeDetector()
    return detector.analyze_video(video_path, transcript_segments)

def find_best_clips(video_path: str, transcript_segments: List[TranscriptSegment], 
                   max_clips: int = 5) -> List[HypeMoment]:
    """Find the best clips from video"""
    detector = HypeDetector()
    moments = detector.analyze_video(video_path, transcript_segments)
    
    # Return top clips by score
    return sorted(moments, key=lambda m: m.score, reverse=True)[:max_clips]

def analyze_clip_potential(video_path: str, transcript_segments: List[TranscriptSegment]) -> Dict:
    """Analyze the clip potential of a video"""
    detector = HypeDetector()
    moments = detector.analyze_video(video_path, transcript_segments)
    return detector.get_detection_stats(moments)