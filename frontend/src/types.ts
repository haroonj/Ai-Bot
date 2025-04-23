export interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
}