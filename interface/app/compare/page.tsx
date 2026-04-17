import { CompareWorkbench } from "@/components/dashboard/compare-workbench";

export default async function ComparePage({
  searchParams,
}: {
  searchParams: Promise<{
    left?: string;
    right?: string;
    with?: string;
  }>;
}) {
  const query = await searchParams;
  const left = query.left;
  const right = query.right ?? query.with;

  return <CompareWorkbench initialLeft={left} initialRight={right} />;
}
