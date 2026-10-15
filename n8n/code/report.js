// n8n Code node: "Run report" - pass the join report on as its own item.
return [{ json: $input.first().json.report }];
