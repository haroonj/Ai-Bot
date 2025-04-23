// frontend/src/components/ChatMessageList.tsx
import React, { useRef, useEffect } from 'react';
import { List, ListItem, Paper, Typography, Box } from '@mui/material';
import { ChatMessage } from '../types'; // Import the type

interface ChatMessageListProps {
  messages: ChatMessage[];
}

const ChatMessageList: React.FC<ChatMessageListProps> = ({ messages }) => {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]); // Scroll when messages change

  return (
    <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 2 }}>
      <List sx={{ py: 0 }}>
        {messages.map((message) => (
          <ListItem key={message.id} sx={{ py: 0.5, justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start' }}>
            <Paper
              elevation={2}
              sx={{
                p: 1.5,
                borderRadius: message.role === 'user' ? '15px 15px 0 15px' : '15px 15px 15px 0',
                bgcolor: message.role === 'user' ? 'primary.light' : 'grey.200',
                color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
                maxWidth: '75%',
                wordWrap: 'break-word',
              }}
            >
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}> {/* Preserve whitespace/newlines */}
                {message.content}
              </Typography>
            </Paper>
          </ListItem>
        ))}
        {/* Invisible div to target for scrolling */}
        <div ref={messagesEndRef} />
      </List>
    </Box>
  );
};

export default ChatMessageList;