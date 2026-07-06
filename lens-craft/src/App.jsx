import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout.jsx';
import Landing from './pages/Landing.jsx';
import ModuleRoute from './pages/ModuleRoute.jsx';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Landing />} />
        <Route path="/learn/:slug" element={<ModuleRoute />} />
      </Route>
    </Routes>
  );
}
