import { Link } from 'react-router-dom';

export function Header() {
  return (
    <header className="fixed left-0 top-0 z-20 w-full border-b border-hairline bg-bg/90 backdrop-blur">
      <div className="mx-auto flex max-w-4xl items-center px-5 py-3.5">
        <Link
          to="/"
          className="font-display text-[17px] font-semibold tracking-tight text-text transition-colors hover:text-accent"
        >
          Mythcraft
        </Link>
      </div>
    </header>
  );
}

export default Header;
