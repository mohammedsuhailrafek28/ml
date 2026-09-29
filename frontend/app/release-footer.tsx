'use client';

import {useEffect, useState} from 'react';

type ReleaseInfo = {
  version: string;
  release_status: string;
  source_revision: string;
  model_release: {identifier: string; manifest_sha256: string};
};

export default function ReleaseFooter() {
  const [release, setRelease] = useState<ReleaseInfo | null>(null);
  useEffect(() => {
    let active = true;
    fetch('/api/health', {cache: 'no-store'})
      .then((response) => response.ok ? response.json() : null)
      .then((data) => { if (active && data?.version) setRelease(data); })
      .catch(() => undefined);
    return () => { active = false; };
  }, []);
  return <footer>
    Educational use only — not a medical diagnosis.
    {release && <details className="release-info"><summary>{release.version} · {release.release_status}</summary>
      <span>Source revision: {release.source_revision}; model release: {release.model_release.identifier}; manifest SHA-256: {release.model_release.manifest_sha256}</span>
    </details>}
  </footer>;
}
