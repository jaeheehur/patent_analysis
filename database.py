from sqlalchemy import Column, Integer, String, Date, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import create_engine
import datetime

Base = declarative_base()

class Keyword(Base):
    __tablename__ = 'keywords'
    id = Column(Integer, primary_key=True)
    text = Column(String(255), unique=True, nullable=False)
    source = Column(String(50)) # 'manual' or 'auto'
    last_processed = Column(DateTime)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Patent(Base):
    __tablename__ = 'patents'
    id = Column(Integer, primary_key=True)
    application_number = Column(String(50), unique=True, nullable=False)
    patent_number = Column(String(50))
    title = Column(Text)
    abstract = Column(Text)
    claims = Column(Text)
    applicant = Column(String(255))
    filing_date = Column(Date)
    source_file = Column(String(255)) # Path to uploaded Wipson file
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

engine = create_engine('sqlite:///data/patent_platform.db')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
