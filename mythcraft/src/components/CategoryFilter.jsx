import { CATEGORIES } from '../data/myths';

export function CategoryFilter({ active, onChange }) {
  return (
    <nav aria-label="Filter myths by category">
      <ul className="flex flex-wrap gap-x-5 gap-y-2.5">
        {CATEGORIES.map((category) => {
          const isActive = category === active;
          return (
            <li key={category}>
              <button
                type="button"
                onClick={() => onChange(category)}
                aria-pressed={isActive}
                className={`-mb-px border-b pb-1.5 text-[14px] transition-colors ${
                  isActive
                    ? 'border-accent text-accent'
                    : 'border-transparent text-textDim hover:text-text'
                }`}
              >
                {category}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

export default CategoryFilter;
