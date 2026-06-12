import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "./api";

interface ProjectState {
  projects: string[];
  project: string;
  setProject: (pid: string) => void;
}

const ProjectContext = createContext<ProjectState>({
  projects: [],
  project: "demo",
  setProject: () => undefined,
});

/** Loads the project list once and exposes the current selection. */
export function ProjectProvider({ children }: { children: ReactNode }) {
  const [projects, setProjects] = useState<string[]>([]);
  const [project, setProject] = useState("demo");

  useEffect(() => {
    api
      .projects()
      .then((ids) => {
        setProjects(ids);
        if (ids.length && !ids.includes("demo")) setProject(ids[0]);
      })
      .catch(() => setProjects([]));
  }, []);

  return (
    <ProjectContext.Provider value={{ projects, project, setProject }}>
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject(): ProjectState {
  return useContext(ProjectContext);
}
