type InsightCardProps = {
  title: string;
  body: string;
};

export function InsightCard({ title, body }: InsightCardProps) {
  return (
    <article className="insight-card-web">
      <span>{title}</span>
      <p>{body}</p>
    </article>
  );
}
