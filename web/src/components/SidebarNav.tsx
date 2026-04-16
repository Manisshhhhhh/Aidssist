type ViewName = "upload" | "explore" | "analyze" | "history";

type SidebarNavProps = {
  activeView: ViewName;
  onSelect: (view: ViewName) => void;
  userName: string;
};

const items: Array<{ id: ViewName; label: string; description: string }> = [
  { id: "upload", label: "Upload", description: "Bring in a dataset" },
  { id: "explore", label: "Explore", description: "Understand the signal" },
  { id: "analyze", label: "Analyze", description: "Ask decision questions" },
  { id: "history", label: "History", description: "Review prior runs" }
];

export function SidebarNav({ activeView, onSelect, userName }: SidebarNavProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <div className="sidebar__eyebrow">Aidssist SaaS</div>
        <h1>AI Data Workspace</h1>
        <p>Structured for product use, not just one-off analysis.</p>
      </div>

      <div className="sidebar__user">
        <span>Signed in</span>
        <strong>{userName}</strong>
      </div>

      <nav className="sidebar__nav">
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`sidebar__item ${activeView === item.id ? "sidebar__item--active" : ""}`}
            onClick={() => onSelect(item.id)}
          >
            <span>{item.label}</span>
            <small>{item.description}</small>
          </button>
        ))}
      </nav>
    </aside>
  );
}
