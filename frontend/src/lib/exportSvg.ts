/**
 * Export an SVG element to a downloadable file.
 * Uses standard Web APIs: XMLSerializer, Blob, URL.createObjectURL
 */
export function exportSvgToFile(svgElement: SVGSVGElement, filename: string): void {
  // Clone SVG to avoid modifying the original
  const clone = svgElement.cloneNode(true) as SVGSVGElement;

  // Ensure proper namespace for standalone SVG
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');

  // Calculate viewBox from content bounds
  const bbox = svgElement.getBBox();
  if (bbox.width > 0 && bbox.height > 0) {
    // Add padding around content
    const padding = 20;
    clone.setAttribute(
      'viewBox',
      `${bbox.x - padding} ${bbox.y - padding} ${bbox.width + padding * 2} ${bbox.height + padding * 2}`
    );
    clone.setAttribute('width', String(bbox.width + padding * 2));
    clone.setAttribute('height', String(bbox.height + padding * 2));
  }

  // Serialize to string
  const serializer = new XMLSerializer();
  const svgString = serializer.serializeToString(clone);

  // Create blob and trigger download
  const blob = new Blob([svgString], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);

  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();

  URL.revokeObjectURL(url);
}

/**
 * Generate a filename with timestamp for exported SVG
 */
export function generateSvgFilename(prefix: string): string {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  return `${prefix}-${timestamp}.svg`;
}