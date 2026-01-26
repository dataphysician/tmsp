/**
 * Text wrapping and formatting utilities for GraphViewer.
 */

/**
 * Debounce helper for resize handling.
 * Delays execution until after the specified wait time has passed since the last call.
 */
export function debounce<T extends (...args: unknown[]) => void>(fn: T, ms: number): T {
  let timeoutId: ReturnType<typeof setTimeout>;
  return ((...args: unknown[]) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), ms);
  }) as T;
}

/**
 * Wrap text into multiple lines with word boundaries.
 * Used for generic text wrapping.
 *
 * @param text - The text to wrap
 * @param maxWidth - Maximum characters per line
 * @returns Array of lines
 */
export function wrapText(text: string, maxWidth: number): string[] {
  if (!text) return [''];
  const words = text.split(' ');
  const lines: string[] = [];
  let currentLine = '';

  for (const word of words) {
    const testLine = currentLine ? currentLine + ' ' + word : word;
    if (testLine.length <= maxWidth) {
      currentLine = testLine;
    } else {
      if (currentLine) lines.push(currentLine);
      currentLine = word;
    }
  }
  if (currentLine) lines.push(currentLine);
  return lines.length ? lines : [''];
}

/**
 * Wrap node label into exactly two lines with word boundaries.
 * Adds ellipsis if content overflows the two lines.
 *
 * @param label - The node label to wrap
 * @param maxCharsPerLine - Maximum characters per line
 * @returns Tuple of [line1, line2]
 */
export function wrapNodeLabel(
  label: string | undefined,
  maxCharsPerLine: number
): [string, string] {
  if (!label) return ['', ''];

  const words = label.split(' ');
  let line1 = '';
  let line2 = '';
  let wordIndex = 0;

  // Build first line - fit complete words only
  for (; wordIndex < words.length; wordIndex++) {
    const word = words[wordIndex];
    const testLine = line1 ? line1 + ' ' + word : word;
    if (testLine.length <= maxCharsPerLine) {
      line1 = testLine;
    } else {
      break;
    }
  }

  // If no words fit on line1, put the first word (even if too long)
  if (!line1 && words.length > 0) {
    line1 = words[0].length > maxCharsPerLine
      ? words[0].substring(0, maxCharsPerLine - 1) + '…'
      : words[0];
    wordIndex = 1;
  }

  // Build second line from remaining words
  for (; wordIndex < words.length; wordIndex++) {
    const word = words[wordIndex];
    const testLine = line2 ? line2 + ' ' + word : word;
    if (testLine.length <= maxCharsPerLine - 1) { // Reserve space for potential ellipsis
      line2 = testLine;
    } else {
      // More words remain - add ellipsis
      if (line2) {
        line2 = line2 + '…';
      } else {
        // First word of line2 is too long
        line2 = word.substring(0, maxCharsPerLine - 1) + '…';
      }
      return [line1, line2];
    }
  }

  // Check if there were more words that didn't fit
  if (wordIndex < words.length && line2 && !line2.endsWith('…')) {
    line2 = line2 + '…';
  }

  return [line1, line2];
}

/**
 * Format elapsed time as human-readable string.
 *
 * @param ms - Elapsed time in milliseconds
 * @returns Formatted string (e.g., "5.2s" or "1m 30.5s")
 */
export function formatElapsedTime(ms: number): string {
  const seconds = ms / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
}
