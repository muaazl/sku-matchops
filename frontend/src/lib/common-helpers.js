/**
 * Builds a full path to a static asset file, taking into account the Vite base URL.
 * @param {string} filePath - The relative file path (e.g., 'logo.svg').
 * @return {string} - The full asset path with the base URL prepended.
 */
export const getFullPath = (filePath) => {
  const baseUrl = import.meta.env.BASE_URL || '/';
  const normalizedBase = baseUrl.replace(/\/?$/, '/');
  const normalizedPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
  return `${normalizedBase}${normalizedPath}`;
};
