"use client";

import OrganizationsTable from "./_components/organizations";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function OrganizationsPage() {
  const { accessToken, userRole } = useAuthorized();
  return <OrganizationsTable userRole={userRole ?? ""} accessToken={accessToken} />;
}
