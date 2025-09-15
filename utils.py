import logging

# Configure logging
logging.basicConfig(
    filename="logs/activity.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - User=%(user)s Action=%(action)s Details=%(details)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Add filter for structured logging
class ContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "user"):
            record.user = "unknown"
        if not hasattr(record, "action"):
            record.action = "unknown"
        if not hasattr(record, "details"):
            record.details = "-"
        return True

logger = logging.getLogger()
logger.addFilter(ContextFilter())

# Function to log actions
def log_action(user, action, details=None):
    logger.info("", extra={"user": user, "action": action, "details": details or "-"})
