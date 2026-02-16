import os
import datetime
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from database import SessionLocal, Keyword, init_db
from sqlalchemy import func

# Configure logging
logging.basicConfig(level=logging.INFO, filename='logs/scheduler.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("patent-scheduler")

KEYWORD_FILE = "queries/keyword.txt"

def process_keywords():
    logger.info("Starting weekly keyword processing...")
    db = SessionLocal()
    try:
        # Load from file
        if os.path.exists(KEYWORD_FILE):
            with open(KEYWORD_FILE, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            for text in lines:
                existing = db.query(Keyword).filter(Keyword.text == text).first()
                if not existing:
                    new_kw = Keyword(text=text, source='auto')
                    db.add(new_kw)
                else:
                    existing.last_processed = datetime.datetime.utcnow()
            
            db.commit()
            logger.info(f"Processed {len(lines)} keywords from {KEYWORD_FILE}")
    except Exception as e:
        logger.error(f"Error in scheduler: {e}")
        db.rollback()
    finally:
        db.close()

def start_scheduler():
    init_db()
    scheduler = BackgroundScheduler()
    # Schedule weekly on Monday at 00:00
    scheduler.add_job(process_keywords, 'cron', day_of_week='mon', hour=0, minute=0)
    scheduler.start()
    logger.info("Scheduler started.")
    return scheduler

if __name__ == "__main__":
    # Manual trigger for testing
    start_scheduler()
    process_keywords()
