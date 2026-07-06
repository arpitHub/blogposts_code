import { useParams, Navigate } from 'react-router-dom';
import { getModule } from '../data/modules.js';
import ModulePage from '../components/ModulePage.jsx';
import PlaceholderPanel from '../components/PlaceholderPanel.jsx';
import ExposureTriangleModule from '../widgets/exposure-triangle/ExposureTriangleModule.jsx';

// Maps a module slug to its fully-built module page component.
// Anything not in this registry falls back to the placeholder.
const BUILT_MODULES = {
  'exposure-triangle': ExposureTriangleModule,
};

export default function ModuleRoute() {
  const { slug } = useParams();
  const module = getModule(slug);

  if (!module) return <Navigate to="/" replace />;

  const Built = BUILT_MODULES[slug];
  if (Built) return <Built module={module} />;

  return (
    <ModulePage
      module={module}
      intro="This module is on the roadmap and hasn't been built yet."
    >
      <PlaceholderPanel title={module.title} />
    </ModulePage>
  );
}
