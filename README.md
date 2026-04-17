# Aidssist

AI-powered data analytics platform that allows users to upload datasets and interact with them using natural language.

## Features
- Folder upload system
- AI-powered chat for querying data
- Data intelligence dashboard
- Schema detection and processing
- Background task processing using Celery

## Tech Stack
- Backend: FastAPI
- Frontend: React + Vite
- Task Queue: Celery
- Language: Python, TypeScript

## Project Structure
backend/ → API, AI logic, task handling  
web/ → frontend UI  

## How to Run

### Backend
cd backend  
pip install -r requirements.txt  
uvicorn main:app --reload  

### Frontend
cd web  
npm install  
npm run dev  

## Status
Work in progress (actively improving features and UI)