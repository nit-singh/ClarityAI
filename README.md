# AI Lecture Summarizer

A powerful web application that uses AI to automatically summarize lecture videos into structured notes, study plans, and downloadable reports.

## ✨ Features

- **AI-Powered Summarization**: Uses Gemini AI to create comprehensive lecture summaries
- **Multiple Export Formats**: Download summaries as Word (.docx) or LaTeX (.tex) files
- **User Authentication**: Sign up/login system with MongoDB integration
- **Profile Management**: Save preferences and track upload history
- **Guest Mode**: Try the app without creating an account
- **Modern UI/UX**: Beautiful dark theme with neon accents and animations
- **Batch Processing**: Upload multiple videos at once
- **Real-time Progress**: Visual progress tracking during processing

## 🚀 Quick Start

### Prerequisites

- Node.js (v14 or higher)
- Python (v3.8 or higher)
- MongoDB (local or cloud instance)
- Gemini API key (optional - can be set by users)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-lecture-summarizer
   ```

2. **Install dependencies**
   ```bash
   npm install
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   MONGODB_URI=mongodb://localhost:27017/lecture-summarizer
   JWT_SECRET=your-super-secret-jwt-key
   GEMINI_API_KEY=your-gemini-api-key
   ```

4. **Start the application**
   ```bash
   npm start
   ```

5. **Open your browser**
   Navigate to `http://localhost:3000`

## 🔐 Authentication

The application supports three modes:

1. **Registered Users**: Full access with saved preferences and upload history
2. **Guest Mode**: Try the app without creating an account
3. **Anonymous**: Limited functionality

### User Features

- **Profile Management**: Update name, preferences, and API keys
- **Upload History**: Track all processed videos
- **Settings Sync**: Preferences saved across sessions
- **Personal Dashboard**: View statistics and recent activity

## 🎯 Usage

1. **Sign Up/Login** or continue as guest
2. **Upload Videos**: Drag and drop or click to select video files
3. **Configure Settings**: Choose model size and API key (optional)
4. **Process**: Click "Summarize Videos" to start processing
5. **Download**: Get your summaries in Word or LaTeX format
6. **Chat**: Ask questions about your lecture content

## 🛠️ Technical Details

### Backend
- **Node.js/Express**: Web server and API
- **MongoDB**: User data and upload history
- **JWT**: Authentication tokens
- **Multer**: File upload handling

### Frontend
- **Vanilla JavaScript**: No frameworks, pure JS
- **Modern CSS**: CSS Grid, Flexbox, animations
- **Responsive Design**: Works on desktop and mobile

### AI Processing
- **Python**: Video processing and AI integration
- **Gemini API**: Text summarization
- **Whisper**: Audio transcription
- **Multiple Models**: Support for different model sizes

## 📁 Project Structure

```
├── public/                 # Frontend files
│   ├── index.html         # Main application
│   ├── login.html         # Login page
│   ├── signup.html        # Signup page
│   ├── styles.css         # Main styles
│   ├── auth.css          # Authentication styles
│   ├── main.js           # Main JavaScript
│   └── auth.js           # Authentication JavaScript
├── server/                # Backend files
│   ├── index.js          # Express server
│   └── models/           # Database models
│       └── User.js       # User schema
├── src/                   # AI processing (unchanged)
│   └── ai_lecture_summarizer/
├── uploads/               # Uploaded videos
├── outputs/              # Generated summaries
└── data/                 # Application data
```

## 🔧 Configuration

### Environment Variables

- `MONGODB_URI`: MongoDB connection string
- `JWT_SECRET`: Secret key for JWT tokens
- `GEMINI_API_KEY`: Default Gemini API key
- `PYTHON_BIN`: Python executable path
- `PORT`: Server port (default: 3000)

### Model Sizes

- **tiny**: Fastest, least accurate
- **base**: Balanced (recommended)
- **small**: Better quality
- **medium**: High quality
- **large-v3**: Best quality, slowest

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with details
4. Include error logs and steps to reproduce
