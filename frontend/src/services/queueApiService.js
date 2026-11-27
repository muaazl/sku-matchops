import { httpClient } from './httpClient';

export async function getDeadLetters() {
  return (await httpClient.get('/queue/dlq')).data;
}

export async function replayDeadLetter(taskId) {
  return (await httpClient.post(`/queue/dlq/${taskId}/replay`)).data;
}
