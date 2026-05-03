export function sanitizeFavoriteWorks(value: unknown): string[] {
  if (!Array.isArray(value)) return [];

  const result: string[] = [];
  const seen = new Set<string>();

  for (const item of value) {
    if (typeof item !== "string") continue;

    const title = item.trim();
    if (!title) continue;

    const key = title.toLocaleLowerCase("pt-BR");
    if (seen.has(key)) continue;

    seen.add(key);
    result.push(title);
  }

  return result.slice(0, 50);
}

export function upsertFavoriteWork(current: unknown, title: string): string[] {
  const cleanTitle = title.trim();
  if (!cleanTitle) return sanitizeFavoriteWorks(current);

  return sanitizeFavoriteWorks([cleanTitle, ...sanitizeFavoriteWorks(current)]);
}

export function getFavoriteWorkSuggestions(
  favoriteWorks: unknown,
  query = "",
  limit = 8,
): string[] {
  const items = sanitizeFavoriteWorks(favoriteWorks);
  const normalizedQuery = query.trim().toLocaleLowerCase("pt-BR");

  if (!normalizedQuery) return items.slice(0, limit);

  return items
    .filter((item) => item.toLocaleLowerCase("pt-BR").includes(normalizedQuery))
    .slice(0, limit);
}
