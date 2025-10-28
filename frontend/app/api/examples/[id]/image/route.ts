export const dynamic = "force-dynamic";

interface Params {
  params: { id: string };
}

export async function GET(_req: Request, { params }: Params): Promise<Response> {
  const backend = process.env.BACKEND_URL;
  if (!backend) {
    return new Response("BACKEND_URL not configured", { status: 503 });
  }
  const id = params.id;
  try {
    const rsp = await fetch(`${backend}/examples/${id}/image.png`);
    const buf = await rsp.arrayBuffer();
    return new Response(buf, {
      status: rsp.status,
      headers: { "content-type": rsp.headers.get("content-type") || "image/png" },
    });
  } catch (err) {
    return new Response("Failed to fetch example image", { status: 502 });
  }
}


