import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class StreamConfig:
    """Configuration for stream processing"""
    # Stream settings
    stream_url: str = "https://www.twitch.tv/yourstreamer"
    platform: str = "twitch"  # "twitch" or "youtube"
    
    # Processing intervals
    segment_duration: int = 300  # 5 minutes in seconds
    processing_interval: int = 180  # Check every 3 minutes
    
    # Clip settings
    min_clip_duration: int = 15  # seconds
    max_clip_duration: int = 60  # seconds
    target_aspect_ratio: str = "9:16"  # TikTok/Shorts format
    
    # Detection settings
    volume_threshold: float = 0.7  # 0-1 scale for volume spike detection
    keyword_boost: float = 1.5  # Multiplier for keyword matches
    
    # Hype keywords (commonly used in gaming/streaming)
    hype_keywords: List[str] = None
    
    # File paths
    output_dir: str = "clips"
    temp_dir: str = "temp"
    assets_dir: str = "assets"
    logs_dir: str = "logs"
    
    # API keys
    openai_api_key: str = ""
    
    # FFmpeg settings
    video_quality: str = "medium"  # low, medium, high
    audio_bitrate: str = "128k"
    video_bitrate: str = "2000k"
    
    def __post_init__(self):
        if self.hype_keywords is None:
            self.hype_keywords = [
                # Gaming reactions
                "oh my god", "no way", "let's go", "what the hell", "holy shit",
                "insane", "crazy", "unbelievable", "amazing", "incredible",
                "clutch", "ez", "gg", "rekt", "destroyed", "owned",
                
                # Streaming reactions
                "chat", "viewers", "subscribe", "follow", "donation",
                "pog", "poggers", "kappa", "lul", "omegalul",
                "nahh", "bruh", "bro", "dude", "guys",
                
                # Excitement words
                "yooo", "wooo", "lets gooo", "sheesh", "fire",
                "sick", "dope", "lit", "based", "cringe",
                
                # Shock/surprise
                "what", "how", "why", "wait", "hold up",
                "pause", "stop", "rewind", "replay"
            ]
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.temp_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)

# Global config instance
config = StreamConfig()

# Environment variable overrides
if os.getenv("OPENAI_API_KEY"):
    config.openai_api_key = os.getenv("OPENAI_API_KEY")

if os.getenv("STREAM_URL"):
    config.stream_url = os.getenv("STREAM_URL")

# Quality presets
QUALITY_PRESETS = {
    "low": {
        "video_bitrate": "1000k",
        "audio_bitrate": "96k",
        "scale": "720:1280"  # 9:16 aspect ratio
    },
    "medium": {
        "video_bitrate": "2000k",
        "audio_bitrate": "128k",
        "scale": "1080:1920"  # 9:16 aspect ratio
    },
    "high": {
        "video_bitrate": "4000k",
        "audio_bitrate": "192k",
        "scale": "1080:1920"  # 9:16 aspect ratio
    }
}

# Platform-specific settings
PLATFORM_CONFIGS = {
    "twitch": {
        "base_url": "https://www.twitch.tv/",
        "api_url": "https://api.twitch.tv/helix/",
        "requires_auth": True
    },
    "youtube": {
        "base_url": "https://www.youtube.com/",
        "api_url": "https://www.googleapis.com/youtube/v3/",
        "requires_auth": False
    }
}

def get_quality_preset(quality: str) -> Dict[str, Any]:
    """Get quality preset by name"""
    return QUALITY_PRESETS.get(quality, QUALITY_PRESETS["medium"])

def update_config(**kwargs):
    """Update configuration values"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config key '{key}'")