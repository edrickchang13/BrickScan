/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./App.tsx', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#FF6B00',
        secondary: '#2D3436',
        accent: '#00B894',
        danger: '#D63031',
      },
    },
  },
  plugins: [],
};
