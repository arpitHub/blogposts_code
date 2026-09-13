import { Navigate, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import Hub from './pages/Hub';
import MythDeepDive from './pages/MythDeepDive';

export function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<Hub />} />
        <Route path="/myth/:slug" element={<MythDeepDive />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

export default App;
