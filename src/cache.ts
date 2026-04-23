export class LRUCache<V> {
  private map = new Map<string, V>();
  private hits = 0;
  private misses = 0;
  constructor(private capacity: number = 100_000) {}

  get(key: string): V | undefined {
    const v = this.map.get(key);
    if (v === undefined) {
      this.misses++;
      return undefined;
    }
    this.map.delete(key);
    this.map.set(key, v);
    this.hits++;
    return v;
  }

  set(key: string, value: V): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.capacity) {
      const firstKey = this.map.keys().next().value;
      if (firstKey !== undefined) this.map.delete(firstKey);
    }
  }

  stats() {
    const total = this.hits + this.misses;
    return {
      size: this.map.size,
      hits: this.hits,
      misses: this.misses,
      hit_rate: total === 0 ? 0 : this.hits / total,
    };
  }
}
