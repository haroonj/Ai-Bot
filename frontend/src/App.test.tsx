import React from 'react';
import {render, screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import ChatInput from "./components/ChatInput";
import {ChatMessage} from "./types";
import ChatMessageList from "./components/ChatMessageList";

// Use the same default URL as in App.tsx for consistency
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/chat';

describe('<App /> Component', () => {
    test('renders the chat interface correctly', () => {
        render(<App/>);
        expect(screen.getByRole('heading', {name: /ai e-commerce support bot/i})).toBeInTheDocument();
        expect(screen.getByPlaceholderText(/ask about orders/i)).toBeInTheDocument();
        expect(screen.getByRole('button', {name: /send/i})).toBeInTheDocument();
        // Send button should be disabled initially
        expect(screen.getByRole('button', {name: /send/i})).toBeDisabled();
    });

    test('allows typing in the input field and enables send button', async () => {
        render(<App/>);
        const input = screen.getByPlaceholderText(/ask about orders/i);
        const sendButton = screen.getByRole('button', {name: /send/i});

        expect(sendButton).toBeDisabled(); // Initially disabled

        await userEvent.type(input, 'Hello bot');

        expect(input).toHaveValue('Hello bot');
        expect(sendButton).toBeEnabled(); // Enabled after typing
    });

    test('sends a message and displays user/bot responses on success', async () => {
        render(<App/>);
        const input = screen.getByPlaceholderText(/ask about orders/i);
        const sendButton = screen.getByRole('button', {name: /send/i});

        // Type a message and send
        await userEvent.type(input, 'Status of ORD123');
        await userEvent.click(sendButton);

        // Check if user message appears
        expect(screen.getByText('Status of ORD123')).toBeInTheDocument();

        // Check if input is cleared
        expect(input).toHaveValue('');

        // Check if loading state is active (button disabled)
        expect(sendButton).toBeDisabled();
        // You could also check for the spinner if it had a specific role/testid

        // Wait for the mocked bot response to appear
        // Use findByText which incorporates waitFor
        const botResponse = await screen.findByText(/Mock response: The status for order ORD123 is: Shipped./i);
        expect(botResponse).toBeInTheDocument();

        // Check if loading state is finished (button enabled if input empty, or disabled if empty)
        expect(sendButton).toBeDisabled(); // Disabled because input is empty
    });

    test('handles API error and displays snackbar', async () => {
        render(<App/>);
        const input = screen.getByPlaceholderText(/ask about orders/i);
        const sendButton = screen.getByRole('button', {name: /send/i});

        // Type message that triggers error in mock handler
        await userEvent.type(input, 'Trigger an error');
        await userEvent.click(sendButton);

        // Check user message still appears
        expect(screen.getByText('Trigger an error')).toBeInTheDocument();
        // Check input is cleared
        expect(input).toHaveValue('');

        // Wait for the error Snackbar alert to appear
        const errorAlert = await screen.findByRole('alert');
        expect(errorAlert).toBeInTheDocument();
        expect(errorAlert).toHaveTextContent(/Mock server error occurred!/i);

        // Check if loading state is finished
        expect(sendButton).toBeDisabled(); // Disabled because input is empty
    });

    test('does not send empty messages', async () => {
        render(<App/>);
        const input = screen.getByPlaceholderText(/ask about orders/i);
        const sendButton = screen.getByRole('button', {name: /send/i});

        // Try typing only whitespace
        await userEvent.type(input, '   ');
        expect(sendButton).toBeDisabled();

        // Clear input
        await userEvent.clear(input);
        expect(sendButton).toBeDisabled();

        // Try clicking send button while disabled
        await userEvent.click(sendButton);

        // Assert no messages were added (initial state)
        const messages = screen.queryAllByRole('listitem'); // Find list items which contain messages
        expect(messages).toHaveLength(0);
    });

    // Optional: Test sending with Enter key
    test('sends message on Enter key press', async () => {
        render(<App/>);
        const input = screen.getByPlaceholderText(/ask about orders/i);

        await userEvent.type(input, 'Test Enter Key{enter}');

        // Check if user message appears
        expect(screen.getByText('Test Enter Key')).toBeInTheDocument();
        // Check if input is cleared
        expect(input).toHaveValue('');
        // Wait for the default mocked bot response
        expect(await screen.findByText(/Mock response for: "Test Enter Key"/i)).toBeInTheDocument();
    });

});

// You can add more specific tests for ChatInput and ChatMessageList if desired
// Example (add to this file or create separate test files):

describe('<ChatInput />', () => {
    test('calls onSendMessage when send button is clicked', async () => {
        const handleSendMock = jest.fn();
        render(<ChatInput onSendMessage={handleSendMock} isLoading={false}/>);
        const input = screen.getByPlaceholderText(/ask about orders/i);
        const sendButton = screen.getByRole('button', {name: /send/i});

        await userEvent.type(input, 'My Test Message');
        await userEvent.click(sendButton);

        expect(handleSendMock).toHaveBeenCalledTimes(1);
        expect(handleSendMock).toHaveBeenCalledWith('My Test Message');
    });
});

describe('<ChatMessageList />', () => {
    test('renders user and assistant messages', () => {
        const messages: ChatMessage[] = [{id: 1, role: 'user', content: 'Hello'}, {
            id: 2,
            role: 'assistant',
            content: 'Hi there!'
        }];
        render(<ChatMessageList messages={messages}/>);
        expect(screen.getByText('Hello')).toBeInTheDocument();
        expect(screen.getByText('Hi there!')).toBeInTheDocument();
    });
});
