// frontend/src/components/ChatInput.tsx
import React, { useState } from 'react';
import { TextField, IconButton, Box, CircularProgress } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isLoading }) => {
  const [inputValue, setInputValue] = useState('');

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  };

  const handleSendClick = () => {
    const trimmedValue = inputValue.trim();
    if (trimmedValue && !isLoading) {
      onSendMessage(trimmedValue);
      setInputValue(''); // Clear input after sending
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    // Send on Enter, but allow Shift+Enter for new lines
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault(); // Prevent default newline behavior
      handleSendClick();
    }
  };

  return (
    <Box
      sx={{
        p: 2,
        borderTop: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
        display: 'flex',
        alignItems: 'center',
        gap: 1, // Add gap between text field and button
      }}
    >
      <TextField
        fullWidth
        multiline
        maxRows={5} // Allow up to 5 rows before scrolling
        variant="outlined"
        size="small" // Make it a bit smaller
        placeholder="Ask about orders, tracking, returns..."
        value={inputValue}
        onChange={handleInputChange}
        onKeyPress={handleKeyPress}
        disabled={isLoading}
        sx={{
          '& .MuiOutlinedInput-root': {
            borderRadius: '20px', // Rounded corners
          },
        }}
      />
      <IconButton
        color="primary"
        onClick={handleSendClick}
        disabled={isLoading || !inputValue.trim()}
        sx={{
            position: 'relative', // Needed for spinner positioning
            bgcolor: 'primary.main',
            color: 'primary.contrastText',
            '&:hover': { bgcolor: 'primary.dark'},
            '&.Mui-disabled': { bgcolor: 'action.disabledBackground' }
        }}
      >
        {isLoading ? <CircularProgress size={24} sx={{
            color: 'primary.contrastText',
            position: 'absolute',
            top: '50%',
            left: '50%',
            marginTop: '-12px',
            marginLeft: '-12px',
          }}/> : <SendIcon />}
      </IconButton>
    </Box>
  );
};

export default ChatInput;