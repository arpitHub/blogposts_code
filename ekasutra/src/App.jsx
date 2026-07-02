import { useState } from 'react';
import Landing from './components/Landing.jsx';
import ThemeNavigator from './components/ThemeNavigator.jsx';
import ThemeDetail from './components/ThemeDetail.jsx';
import AskSage from './components/AskSage.jsx';

export default function App() {
  const [view, setView] = useState('landing');
  const [currentThemeId, setCurrentThemeId] = useState(null);

  function openTheme(id) {
    setCurrentThemeId(id);
    setView('detail');
  }

  function goToNavigator() {
    setView('navigator');
  }

  function goHome() {
    setView('landing');
    setCurrentThemeId(null);
  }

  return (
    <div className="app">
      <div className="app-noise" aria-hidden="true" />

      {view === 'landing' && <Landing onEnter={goToNavigator} />}

      {view === 'navigator' && (
        <ThemeNavigator onOpenTheme={openTheme} onBack={goHome} />
      )}

      {view === 'detail' && currentThemeId && (
        <ThemeDetail
          themeId={currentThemeId}
          onSelectTheme={openTheme}
          onBack={goToNavigator}
        />
      )}

      <AskSage />
    </div>
  );
}
