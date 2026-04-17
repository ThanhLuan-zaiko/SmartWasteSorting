import { RunDetailView } from "@/components/dashboard/run-detail-view";
import { SystemNotices } from "@/components/dashboard/system-notices";

export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ experiment: string }>;
}) {
  const { experiment } = await params;

  return (
    <div className="grid gap-6">
      <SystemNotices />
      <RunDetailView experiment={experiment} />
    </div>
  );
}
