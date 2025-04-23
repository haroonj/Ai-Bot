// frontend/src/App.tsx
import React, { useState, useCallback } from 'react';
import {
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Paper,
  Alert,
  Snackbar,
} from '@mui/material';
import ChatMessageList from './components/ChatMessageList';
import ChatInput from './components/ChatInput';
import { ChatMessage } from './types';

// Use environment variable or default to localhost:8000
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/chat';

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  // conversationId state is optional here as backend is stateless
  // const [conversationId, setConversationId] = useState<string | null>(null);

  const handleSendMessage = useCallback(async (query: string) => {
    if (!query || isLoading) return;

    const newUserMessage: ChatMessage = {
      id: Date.now(), // Simple ID generation
      role: 'user',
      content: query,
    };

    // Add user message immediately
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
    setIsLoading(true);
    setError(null); // Clear previous errors

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json', // Important for FastAPI
        },
        body: JSON.stringify({
          query: query,
          // conversation_id: conversationId // Uncomment if backend uses it
        }),
      });

      if (!response.ok) {
        let errorDetail = `HTTP error ${response.status}`;
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorDetail;
        } catch {
          // Ignore if response is not JSON
        }
        throw new Error(errorDetail);
      }

      const data = await response.json();

      const newBotMessage: ChatMessage = {
        id: Date.now() + 1, // Simple unique ID
        role: 'assistant',
        content: data.reply,
      };

      setMessages((prevMessages) => [...prevMessages, newBotMessage]);
      // setConversationId(data.conversation_id); // Uncomment if needed

    } catch (err: any) {
        console.error("API Error:", err);
        const errorMessage = err.message || 'Failed to connect to the bot.';
        setError(errorMessage);
        // Optionally add an error message to the chat history
        // const errorBotMessage: ChatMessage = {
        //   id: Date.now() + 1,
        //   role: 'assistant',
        //   content: `Sorry, an error occurred: ${errorMessage}`
        // };
        // setMessages((prevMessages) => [...prevMessages, errorBotMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]); // Add dependencies if using conversationId

  const handleCloseSnackbar = () => {
    setError(null);
  };

  return (
    <>
      <CssBaseline /> {/* Normalize CSS */}
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              AI E-Commerce Support Bot
            </Typography>
          </Toolbar>
        </AppBar>

        {/* Main chat area */}
        <Container maxWidth="md" sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', py: 2 }}>
          <Paper elevation={3} sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' /* Prevent content spill */ }}>
            {/* Message List takes available space */}
            <ChatMessageList messages={messages} />
            {/* Input area stays at bottom */}
            <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
          </Paper>
        </Container>

        {/* Error Snackbar */}
        <Snackbar
           open={!!error}
           autoHideDuration={6000}
           onClose={handleCloseSnackbar}
           anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
         >
           <Alert onClose={handleCloseSnackbar} severity="error" variant="filled" sx={{ width: '100%' }}>
             {error}
           </Alert>
         </Snackbar>
      </Box>
    </>
  );
}

export default App;