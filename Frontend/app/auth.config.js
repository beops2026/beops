export const authConfig = {
  publicRoutes: [
    '/',
    '/api/load-data',
    '/sign-in(.*)',
    '/sign-up(.*)',
    '/api/auth/(.*)',
  ],
  ignoredRoutes: [
    '/_next/(.*)',
    '/favicon.ico',
    '/static/(.*)',
    '/images/(.*)',
    '/(.*).png',
    '/(.*).jpg',
    '/(.*).svg',
  ],
}; 