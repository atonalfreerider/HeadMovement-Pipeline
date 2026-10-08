import { h, replace } from "../util.js";

export default {
  title: () => "Not found",
  async mount(root) {
    replace(root, h("div", { class: "pad" }, h("h1", null, "Nothing here"),
      h("p", null, "This address is not a monitor view. ", h("a", { href: "#/" }, "Go to the takes"), ".")));
    return {};
  },
};
