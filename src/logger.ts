export function log(event: string, data: Record<string, unknown> = {}): void {
  const entry = { ts: new Date().toISOString(), event, ...data };
  process.stdout.write(JSON.stringify(entry) + "\n");
}
