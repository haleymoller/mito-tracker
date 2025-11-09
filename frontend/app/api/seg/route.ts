export const dynamic = "force-dynamic";

export async function POST(request: Request): Promise<Response> {
  const envBackend = process.env.BACKEND_URL;
  const backend =
    envBackend ?? (process.env.NODE_ENV !== "production" ? "http://127.0.0.1:8000" : undefined);
  if (!backend) {
    return new Response(
      JSON.stringify({ error: "BACKEND_URL is not configured" }),
      { status: 503, headers: { "content-type": "application/json" } }
    );
  }
  try {
    const form = await request.formData();
    const rsp = await fetch(`${backend}/seg`, { method: "POST", body: form });
    const body = await rsp.text();
    return new Response(body, {
      status: rsp.status,
      headers: { "content-type": rsp.headers.get("content-type") || "application/json" },
    });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "Failed to reach backend /seg" }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }
}


