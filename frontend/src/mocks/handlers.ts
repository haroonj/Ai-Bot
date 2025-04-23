import { http, HttpResponse, delay } from 'msw';

// Use the same default URL as in App.tsx for consistency
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/chat';

export const handlers = [
  // Intercept POST requests to the chat endpoint
  http.post(API_URL, async ({ request }) => {
    // Simulate network delay
    await delay(150);

    const requestBody = await request.json() as { query?: string };

    // --- Define Mock Responses Based on Query ---
    if (requestBody.query?.toLowerCase().includes('error')) {
      // Simulate a server error
      return HttpResponse.json(
        { detail: 'Mock server error occurred!' },
        { status: 500 }
      );
    }

    if (requestBody.query?.toLowerCase().includes('status of ord123')) {
       return HttpResponse.json({
         reply: 'Mock response: The status for order ORD123 is: Shipped.',
         conversation_id: 'mock-conv-id-123',
       });
    }

    // Default success response for other queries
    return HttpResponse.json({
      reply: `Mock response for: "${requestBody.query ?? 'empty query'}"`,
      conversation_id: 'mock-conv-id-default',
    });
  }),

  // Add other handlers here if needed (e.g., for GET requests if any)
];
