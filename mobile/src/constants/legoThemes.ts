export interface LegoTheme {
  id: string;
  label: string;
  emoji: string;
}

export const LEGO_THEMES: LegoTheme[] = [
  { id: 'all', label: 'All Themes', emoji: '🧱' },
  { id: '171', label: 'Star Wars', emoji: '⚔️' },
  { id: '52', label: 'Technic', emoji: '⚙️' },
  { id: '49', label: 'City', emoji: '🏙️' },
  { id: '186', label: 'Creator 3-in-1', emoji: '🔄' },
  { id: '1', label: 'Classic', emoji: '🎨' },
  { id: '9', label: 'Duplo', emoji: '👶' },
  { id: '158', label: 'Friends', emoji: '👯' },
  { id: '107', label: 'Ninjago', emoji: '🥋' },
  { id: '126', label: 'Minecraft', emoji: '⛏️' },
  { id: '128', label: 'The Hobbit', emoji: '🧙' },
  { id: '231', label: 'The Lord of the Rings', emoji: '💍' },
  { id: '131', label: 'Agents', emoji: '🕵️' },
  { id: '129', label: 'Monster Fighters', emoji: '👻' },
  { id: '140', label: 'Architecture', emoji: '🏛️' },
  { id: '84', label: 'Sports', emoji: '⚽' },
  { id: '71', label: 'Supplemental', emoji: '📦' },
  { id: '191', label: 'Power Miners', emoji: '⛏️' },
  { id: '192', label: 'Atlantis', emoji: '🌊' },
  { id: '193', label: 'Space Police', emoji: '👮' },
  { id: '194', label: 'Castle', emoji: '🏰' },
  { id: '195', label: 'Pirates', emoji: '🏴‍☠️' },
  { id: '196', label: 'Rescue', emoji: '🚨' },
];

export function getThemeLabel(themeId: string): string {
  const theme = LEGO_THEMES.find((t) => t.id === themeId);
  return theme?.label || 'Unknown';
}

export function getThemeEmoji(themeId: string): string {
  const theme = LEGO_THEMES.find((t) => t.id === themeId);
  return theme?.emoji || '🧱';
}
