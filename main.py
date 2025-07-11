#!/usr/bin/env python3
"""
Auto Clip MVP - Automatically generate viral clips from livestreams
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json
import signal
import threading

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import config, update_config
from downloader.download import StreamDownloader, get_stream_info
from transcriber.transcribe import AudioTranscriber, transcribe_video_file
from detector.detect import HypeDetector, detect_hype_moments
from clipper.clip import VideoClipper, create_all_clips

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = Path(config.logs_dir)
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"auto_clip_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)

class AutoClipProcessor:
    """Main processor for automatic clip generation"""
    
    def __init__(self):
        self.downloader = StreamDownloader()
        self.transcriber = AudioTranscriber()
        self.detector = HypeDetector()
        self.clipper = VideoClipper()
        
        self.session_stats = {
            'segments_processed': 0,
            'clips_created': 0,
            'total_processing_time': 0,
            'start_time': datetime.now()
        }
        
        self.running = False
        self.shutdown_event = threading.Event()
    
    def process_single_segment(self, video_path: str = None, stream_url: str = None) -> List[str]:
        """
        Process a single video segment to create clips
        
        Args:
            video_path: Path to existing video file
            stream_url: URL to download from (if video_path not provided)
            
        Returns:
            List of created clip paths
        """
        start_time = time.time()
        
        try:
            # Step 1: Get video file
            if not video_path:
                if not stream_url:
                    stream_url = config.stream_url
                
                logger.info(f"Downloading segment from {stream_url}")
                video_path = self.downloader.download_segment(stream_url)
                
                if not video_path:
                    logger.error("Failed to download video segment")
                    return []
            
            logger.info(f"Processing video: {video_path}")
            
            # Step 2: Transcribe audio
            logger.info("Transcribing audio...")
            transcript_segments = self.transcriber.transcribe_video(video_path)
            
            if not transcript_segments:
                logger.warning("No transcript generated, proceeding without subtitles")
            else:
                logger.info(f"Generated {len(transcript_segments)} transcript segments")
            
            # Step 3: Detect hype moments
            logger.info("Detecting hype moments...")
            hype_moments = self.detector.analyze_video(video_path, transcript_segments)
            
            if not hype_moments:
                logger.info("No hype moments detected in this segment")
                return []
            
            logger.info(f"Detected {len(hype_moments)} hype moments")
            
            # Step 4: Create clips
            logger.info("Creating clips...")
            created_clips = self.clipper.create_multiple_clips(
                video_path, hype_moments, transcript_segments
            )
            
            # Step 5: Update session stats
            processing_time = time.time() - start_time
            self.session_stats['segments_processed'] += 1
            self.session_stats['clips_created'] += len(created_clips)
            self.session_stats['total_processing_time'] += processing_time
            
            logger.info(f"Created {len(created_clips)} clips in {processing_time:.1f}s")
            
            # Step 6: Log clip details
            self._log_clip_details(created_clips, hype_moments)
            
            return created_clips
            
        except Exception as e:
            logger.error(f"Failed to process segment: {e}")
            return []
    
    def run_continuous_monitoring(self, stream_url: str = None):
        """
        Continuously monitor stream and create clips
        
        Args:
            stream_url: Stream URL to monitor
        """
        if not stream_url:
            stream_url = config.stream_url
        
        logger.info(f"Starting continuous monitoring of {stream_url}")
        logger.info(f"Processing interval: {config.processing_interval} seconds")
        
        # Get stream info
        stream_info = get_stream_info(stream_url)
        if stream_info:
            logger.info(f"Stream: {stream_info.get('title', 'Unknown')} by {stream_info.get('uploader', 'Unknown')}")
            logger.info(f"Live: {stream_info.get('is_live', False)}, Views: {stream_info.get('view_count', 'Unknown')}")
        
        self.running = True
        consecutive_failures = 0
        max_failures = 3
        
        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Check if stream is live (optional safety check)
                    if stream_info and not stream_info.get('is_live', True):
                        logger.warning("Stream appears to be offline, but continuing...")
                    
                    # Process current segment
                    created_clips = self.process_single_segment(stream_url=stream_url)
                    
                    if created_clips:
                        consecutive_failures = 0
                        logger.info(f"Successfully created {len(created_clips)} clips")
                        
                        # Log storage info
                        total_clips = self.session_stats['clips_created']
                        logger.info(f"Total clips created this session: {total_clips}")
                    else:
                        consecutive_failures += 1
                        logger.warning(f"No clips created this cycle (failures: {consecutive_failures})")
                        
                        # Log why no clips were created
                        if consecutive_failures == 1:
                            logger.info("This could be normal - not all segments contain hype moments")
                    
                    # Check if we should stop due to consecutive failures
                    if consecutive_failures >= max_failures:
                        logger.error(f"Too many consecutive failures ({consecutive_failures}), stopping...")
                        break
                    
                    # Wait before next processing cycle
                    if self.running:
                        next_check = datetime.now().strftime('%H:%M:%S')
                        logger.info(f"Waiting {config.processing_interval} seconds before next cycle...")
                        logger.info(f"Next processing cycle scheduled around: {next_check}")
                        if self.shutdown_event.wait(config.processing_interval):
                            break
                    
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt, stopping...")
                    break
                except Exception as e:
                    consecutive_failures += 1
                    logger.error(f"Error in processing cycle: {e}")
                    
                    if consecutive_failures >= max_failures:
                        logger.error(f"Too many consecutive errors ({consecutive_failures}), stopping...")
                        break
                    
                    # Wait a bit before retrying
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping...")
        finally:
            self.running = False
            self._print_session_summary()
    
    def _log_clip_details(self, created_clips: List[str], hype_moments: List):
        """Log details about created clips"""
        if not created_clips:
            return
        
        logger.info("=== CLIP DETAILS ===")
        for i, clip_path in enumerate(created_clips):
            if i < len(hype_moments):
                moment = hype_moments[i]
                logger.info(f"Clip {i+1}: {Path(clip_path).name}")
                logger.info(f"  - Time: {moment.start:.1f}s - {moment.end:.1f}s ({moment.duration:.1f}s)")
                logger.info(f"  - Score: {moment.score:.2f} (Confidence: {moment.confidence:.2f})")
                logger.info(f"  - Triggers: {', '.join(moment.triggers)}")
                if moment.transcript:
                    # Truncate transcript if too long
                    transcript_preview = moment.transcript[:100] + "..." if len(moment.transcript) > 100 else moment.transcript
                    logger.info(f"  - Transcript: {transcript_preview}")
        logger.info("===================")
    
    def _print_session_summary(self):
        """Print summary of the processing session"""
        runtime = datetime.now() - self.session_stats['start_time']
        
        logger.info("=== SESSION SUMMARY ===")
        logger.info(f"Total Runtime: {runtime}")
        logger.info(f"Segments Processed: {self.session_stats['segments_processed']}")
        logger.info(f"Clips Created: {self.session_stats['clips_created']}")
        logger.info(f"Total Processing Time: {self.session_stats['total_processing_time']:.1f}s")
        
        if self.session_stats['segments_processed'] > 0:
            avg_time = self.session_stats['total_processing_time'] / self.session_stats['segments_processed']
            avg_clips = self.session_stats['clips_created'] / self.session_stats['segments_processed']
            logger.info(f"Average Processing Time: {avg_time:.1f}s per segment")
            logger.info(f"Average Clips per Segment: {avg_clips:.1f}")
        
        logger.info("======================")
    
    def stop(self):
        """Stop the continuous monitoring"""
        logger.info("Stopping Auto Clip Processor...")
        self.running = False
        self.shutdown_event.set()


def signal_handler(signum, frame, processor):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    processor.stop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Auto Clip MVP - Generate viral clips from livestreams')
    parser.add_argument('--stream-url', type=str, help='Stream URL to monitor')
    parser.add_argument('--video-path', type=str, help='Path to existing video file to process')
    parser.add_argument('--single', action='store_true', help='Process single segment only')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    # Load custom config if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
            update_config(custom_config)
            logger.info(f"Loaded custom config from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return 1
    
    # Validate configuration
    if not args.video_path and not args.stream_url and not config.stream_url:
        logger.error("No stream URL or video path provided. Use --stream-url or --video-path")
        return 1
    
    # Create processor
    processor = AutoClipProcessor()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler_wrapper(signum, frame):
        signal_handler(signum, frame, processor)
    
    signal.signal(signal.SIGINT, signal_handler_wrapper)
    signal.signal(signal.SIGTERM, signal_handler_wrapper)
    
    try:
        if args.single or args.video_path:
            # Single processing mode
            logger.info("Running in single processing mode")
            created_clips = processor.process_single_segment(
                video_path=args.video_path,
                stream_url=args.stream_url
            )
            
            if created_clips:
                logger.info(f"Successfully created {len(created_clips)} clips:")
                for clip in created_clips:
                    logger.info(f"  - {clip}")
            else:
                logger.info("No clips were created")
                
        elif args.continuous or config.stream_url:
            # Continuous monitoring mode
            logger.info("Running in continuous monitoring mode")
            processor.run_continuous_monitoring(stream_url=args.stream_url)
            
        else:
            logger.error("No processing mode specified. Use --single or --continuous")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        processor.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    logger.info("Auto Clip MVP finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())